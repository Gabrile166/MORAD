"""In-training held-out validation for MORAD (verl `test_freq` equivalent).

Every ``test_freq`` outer steps the engine calls this callback, which:

1. snapshots the live policy weights to a temp checkpoint;
2. runs ``evaluate.py`` in a subprocess on the held-out pool;
3. returns the summary dict so the engine can emit ``val/*`` metrics.

A subprocess is used on purpose: the evaluator builds its own model, oracle
worker and CUDA context, and must not perturb the training process's optimizer,
EMA or RNG state. This mirrors verl's `_validate()` isolation guarantee.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


class PeriodicValidator:
    """Callable that runs a held-out evaluation and returns its summary dict."""

    def __init__(
        self,
        *,
        repo_dir: str | os.PathLike[str],
        config_path: str,
        output_root: str | os.PathLike[str],
        python_executable: str | None = None,
        policy_provider: Any = None,
        target_limit: int | None = None,
        timeout_sec: float = 5400.0,
        env_overrides: Mapping[str, str] | None = None,
        keep_runs: int = 3,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.config_path = config_path
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.python_executable = python_executable or sys.executable
        self.policy_provider = policy_provider
        self.target_limit = target_limit
        self.timeout_sec = float(timeout_sec)
        self.env_overrides = dict(env_overrides or {})
        self.keep_runs = int(keep_runs)
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ helpers
    def _snapshot_policy(self, step: int) -> str | None:
        """Write current policy weights to a temp file for the evaluator."""
        if self.policy_provider is None:
            return None
        import torch

        state = self.policy_provider()
        if state is None:
            return None
        path = self.output_root / f"policy_step{step:06d}.pt"
        torch.save({"model": state}, path)
        return str(path)

    def _prune(self) -> None:
        """Keep only the most recent N validation run dirs to bound disk use."""
        if self.keep_runs <= 0:
            return
        runs = sorted(
            (p for p in self.output_root.glob("step*") if p.is_dir()),
            key=lambda p: p.name,
        )
        for stale in runs[: max(0, len(runs) - self.keep_runs)]:
            for child in sorted(stale.rglob("*"), reverse=True):
                try:
                    child.unlink() if child.is_file() else child.rmdir()
                except OSError:
                    pass
            try:
                stale.rmdir()
            except OSError:
                pass
        for stale_ckpt in sorted(self.output_root.glob("policy_step*.pt"))[: -self.keep_runs]:
            try:
                stale_ckpt.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------- public
    def __call__(self, step: int) -> dict[str, Any] | None:
        out_dir = self.output_root / f"step{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.update(self.env_overrides)
        ckpt = self._snapshot_policy(step)
        if ckpt:
            # evaluate.py loads the policy through the RL config's
            # data.ride_checkpoint, which honours ${RIDE_CHECKPOINT}.
            env["RIDE_CHECKPOINT"] = ckpt
        if self.target_limit is not None:
            env["RIDE_EVAL_TARGET_LIMIT"] = str(int(self.target_limit))

        cmd = [
            self.python_executable,
            "evaluate.py",
            "--config",
            self.config_path,
            "--output-dir",
            str(out_dir),
        ]
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.repo_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            (out_dir / "validation_error.txt").write_text(
                f"timeout after {self.timeout_sec}s\n", encoding="utf-8"
            )
            return None

        (out_dir / "validator_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (out_dir / "validator_stderr.log").write_text(proc.stderr or "", encoding="utf-8")

        summary_path = out_dir / "summary.json"
        if proc.returncode != 0 or not summary_path.exists():
            return None

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["_validation"] = {
            "step": step,
            "elapsed_sec": time.time() - started,
            "output_dir": str(out_dir),
            "policy_checkpoint": ckpt,
        }
        self.history.append(
            {
                "step": step,
                "elapsed_sec": summary["_validation"]["elapsed_sec"],
                "target_count": summary.get("target_count"),
            }
        )
        self._prune()
        return summary
