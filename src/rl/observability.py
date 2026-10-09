"""Small runtime logging and manifest helpers for RIDER RL."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping


def json_safe(value: Any) -> Any:
    """Convert common runtime values to JSON-serializable objects."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().tolist()
        except Exception as exc:
            return {"repr": repr(value), "json_error": repr(exc)}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception as exc:
            return {"repr": repr(value), "json_error": repr(exc)}
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
    return sha256_text(encoded)


def git_state(repo_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Return best-effort git state without failing training outside git."""
    repo = Path(repo_dir)

    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    head = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    diff = run(["git", "diff", "--binary"])
    return {
        "head": head,
        "dirty": bool(status),
        "status": status or "",
        "diff_hash": sha256_text(diff or ""),
    }


def repository_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Best-effort state for local reference repositories used by the RL run."""
    repo = Path(path)
    state: dict[str, Any] = {
        "path": str(repo),
        "exists": repo.exists(),
        "is_git": (repo / ".git").exists(),
    }
    if state["is_git"]:
        state.update(git_state(repo))
    return state


def artifact_hashes(repo_dir: str | os.PathLike[str], resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    """Hash local model/data artifacts that are intentionally ignored by git."""
    repo = Path(repo_dir)
    data = resolved_config.get("data", {})
    reward = resolved_config.get("reward", {})
    oracle = reward.get("oracle", {}) if isinstance(reward, Mapping) else {}
    candidates = {
        "pool_manifest": data.get("pool_manifest") if isinstance(data, Mapping) else None,
        "ride_checkpoint": data.get("ride_checkpoint") if isinstance(data, Mapping) else None,
        "rhofold_checkpoint": oracle.get("ckpt") if isinstance(oracle, Mapping) else None,
    }
    hashes: dict[str, Any] = {}
    for name, raw_path in candidates.items():
        if not raw_path:
            continue
        path = Path(str(raw_path))
        resolved = path if path.is_absolute() else repo / path
        entry: dict[str, Any] = {"path": str(path), "exists": resolved.exists()}
        if resolved.is_file():
            entry["sha256"] = sha256_file(resolved)
        hashes[name] = entry
    return hashes


class JsonlLogger:
    """Append-only local logger with optional W&B mirroring."""

    def __init__(self, path: str | os.PathLike[str], wandb_run: Any = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Log a flat dict of scalar metrics.

        The DDP engine calls this once per optimizer update with the aggregated
        batch metrics (reward, loss components, KL, LR, timing). One JSONL line
        is written and the same dict is forwarded to wandb under the ``metrics/``
        event so it shows up as a separate panel group from the raw step records.
        """
        record = {"event": "metrics", "step": step, **dict(metrics)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(record), sort_keys=True, ensure_ascii=False) + "\n")
        if self.wandb_run is not None:
            wandb_payload = {f"metrics/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            if wandb_payload:
                self.wandb_run.log(wandb_payload, step=step)

    def log(self, event: str, payload: Mapping[str, Any] | None = None, step: int | None = None) -> None:
        record: MutableMapping[str, Any] = {
            "time": time.time(),
            "event": event,
        }
        if step is not None:
            record["step"] = int(step)
        if payload:
            record.update(json_safe(payload))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        if self.wandb_run is not None:
            wandb_payload = {f"rl/{event}/{k}": v for k, v in json_safe(payload or {}).items()}
            if wandb_payload:
                self.wandb_run.log(wandb_payload, step=step)


def maybe_init_wandb(logging_config: Mapping[str, Any] | None, resolved_config: Mapping[str, Any]) -> Any:
    """Return an initialized W&B run only when explicitly enabled."""
    cfg = dict(logging_config or {})
    wandb_cfg = dict(cfg.get("wandb", {})) if isinstance(cfg.get("wandb", {}), Mapping) else {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("logging.wandb.enabled=true but wandb is not installed; install wandb or disable logging.wandb.enabled") from exc
    init_kwargs = {k: v for k, v in wandb_cfg.items() if k != "enabled"}
    init_kwargs.setdefault("config", json_safe(resolved_config))
    return wandb.init(**init_kwargs)


def build_run_manifest(
    repo_dir: str | os.PathLike[str],
    resolved_config: Mapping[str, Any],
    command: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build version evidence saved beside checkpoints."""
    repo = Path(repo_dir)
    manifest = {
        "created_at": time.time(),
        "command": command or sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "repo": git_state(repo),
        "reference_repositories": {
            "DiffusionNFT": repository_state(repo.parent / "third_party" / "DiffusionNFT"),
            "verl": repository_state(repo.parent / "third_party" / "verl"),
            "RhoFold": repository_state(repo.parent / "third_party" / "RhoFold"),
            "rhofold_protocol": repository_state(repo.parent / "third_party" / "rhofold_protocol"),
            "RiboDiffusion": repository_state(repo.parent / "third_party" / "RiboDiffusion"),
            "USalign": repository_state(repo.parent / "third_party" / "USalign"),
        },
        "resolved_config_hash": sha256_json(resolved_config),
        "resolved_config": json_safe(resolved_config),
        "artifact_hashes": artifact_hashes(repo, resolved_config),
    }
    try:
        import torch

        manifest["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        manifest["torch"] = {"error": repr(exc)}
    if extra:
        manifest.update(json_safe(extra))
    return manifest
