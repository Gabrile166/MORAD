"""Atomic checkpointing for RIDER RL runs."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .observability import json_safe, sha256_json


@dataclass
class CheckpointCursor:
    outer_step: int = 0
    optimizer_step: int = 0
    target_cursor: int = 0
    oracle_calls: int = 0


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_normalize_rng_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([_normalize_rng_tensor(item) for item in state["torch_cuda"]])


def _normalize_rng_tensor(value: Any) -> torch.ByteTensor:
    """PyTorch RNG restore APIs require detached CPU uint8 tensors."""

    tensor = torch.as_tensor(value)
    return tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _state_dict(obj: Any) -> Any:
    return obj.state_dict() if obj is not None and hasattr(obj, "state_dict") else None


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    current_policy: torch.nn.Module,
    old_policy: torch.nn.Module | None,
    reference_policy: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any = None,
    amp_scaler: Any = None,
    cursor: CheckpointCursor | Mapping[str, Any] | None = None,
    resolved_config: Mapping[str, Any] | None = None,
    reward_scale_state: Mapping[str, Any] | None = None,
    temperature_state: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    """Write a full checkpoint atomically."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config = json_safe(resolved_config or {})
    payload = {
        "current_policy": current_policy.state_dict(),
        "old_policy": _state_dict(old_policy),
        "reference_policy": _state_dict(reference_policy),
        "optimizer": _state_dict(optimizer),
        "scheduler": _state_dict(scheduler),
        "amp_scaler": _state_dict(amp_scaler),
        "cursor": json_safe(cursor or CheckpointCursor()),
        "reward_scale_state": json_safe(reward_scale_state or {}),
        "temperature_state": json_safe(temperature_state or {}),
        "rng_state": capture_rng_state(),
        "resolved_config": config,
        "resolved_config_hash": sha256_json(config),
        "manifest": json_safe(manifest or {}),
        "extra_state": json_safe(extra_state or {}),
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    current_policy: torch.nn.Module | None = None,
    old_policy: torch.nn.Module | None = None,
    reference_policy: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    amp_scaler: Any = None,
    expected_config: Mapping[str, Any] | None = None,
    strict_config: bool = True,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore attached objects."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if expected_config is not None and strict_config:
        expected_hash = sha256_json(expected_config)
        found_hash = payload.get("resolved_config_hash")
        if found_hash != expected_hash:
            raise ValueError(
                "Checkpoint config hash mismatch: "
                f"expected {expected_hash}, found {found_hash}"
            )
    if current_policy is not None:
        current_policy.load_state_dict(payload["current_policy"])
    if old_policy is not None and payload.get("old_policy") is not None:
        old_policy.load_state_dict(payload["old_policy"])
    if reference_policy is not None and payload.get("reference_policy") is not None:
        reference_policy.load_state_dict(payload["reference_policy"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if amp_scaler is not None and payload.get("amp_scaler") is not None:
        amp_scaler.load_state_dict(payload["amp_scaler"])
    restore_rng_state(payload.get("rng_state"))
    return payload


def write_manifest(path: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(manifest), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp_path, manifest_path)
    return manifest_path
