"""Multi-GPU parallel rollout across targets (data parallelism on the target axis).

Profiling of the single-target path showed:

    generate (diffusion, 1 GPU)   5.89s  (53.4%)
    score    (fold, 8 workers)    3.96s  (35.9%)
    train    (backward+step)      1.18s  (10.7%)

`generate` samples `group_size` candidates through a 50-step diffusion loop, all
on one GPU, so 4 of 5 GPUs sit idle during the longest phase. This module runs
one *target* per GPU concurrently, giving an effective batch of
`n_gpus * group_size` samples that is then reduced with gradient accumulation.

Design constraints honoured here:

* GRPO is on-policy, so every replica must sample from the *same* `old_policy`
  snapshot. Replicas are refreshed from the authoritative weights each round.
* Advantages are computed *within* each target's group (that is what GRPO
  normalises over), never across targets, so per-target groups stay intact.
* Targets have different lengths, so batches are never concatenated; each target
  keeps its own condition and contributes an independent backward pass.
"""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import torch


class ParallelRolloutRunner:
    """Runs rollout+scoring for several targets concurrently, one GPU each."""

    def __init__(
        self,
        *,
        devices: Sequence[torch.device | str],
        rollout: Any,
        rewarder: Any,
        condition_factory: Callable[..., Any],
        old_policy: Any,
        condition_noise_scale: float = 0.0,
        seed_base: int = 0,
    ) -> None:
        if not devices:
            raise ValueError("ParallelRolloutRunner requires at least one device")
        self.devices = [torch.device(d) for d in devices]
        self.rollout = rollout
        self.rewarder = rewarder
        self.condition_factory = condition_factory
        self.condition_noise_scale = float(condition_noise_scale)
        self.seed_base = int(seed_base)

        # One policy replica per device. Replica 0 aliases the authoritative
        # module when it already lives on the first device, avoiding a copy.
        self._authoritative = old_policy
        self.replicas: list[Any] = []
        for idx, device in enumerate(self.devices):
            if idx == 0 and _module_device(old_policy) == device:
                self.replicas.append(old_policy)
            else:
                replica = copy.deepcopy(old_policy).to(device)
                replica.eval()
                self.replicas.append(replica)

        self._executor = ThreadPoolExecutor(
            max_workers=len(self.devices), thread_name_prefix="par-rollout"
        )
        self._lock = threading.Lock()

    @property
    def width(self) -> int:
        return len(self.devices)

    def sync_replicas(self) -> None:
        """Copy authoritative weights into every replica (on-policy guarantee)."""
        source = self._authoritative.state_dict()
        for idx, replica in enumerate(self.replicas):
            if replica is self._authoritative:
                continue
            device = self.devices[idx]
            replica.load_state_dict({k: v.to(device) for k, v in source.items()})
            replica.eval()

    def run_batch(
        self,
        targets: Sequence[Any],
        *,
        outer_step: int,
        rollout_state_factory: Callable[[int, torch.device], Any],
    ) -> list[dict[str, Any]]:
        """Roll out and score `targets` concurrently; results keep input order.

        Each entry is ``{"target", "rollout_batch", "scored_batch", "condition",
        "oracle_calls", "device", "error"}``. A failure on one target is captured
        in its own entry instead of aborting the whole batch.
        """
        if not targets:
            return []
        if len(targets) > len(self.devices):
            raise ValueError(
                f"got {len(targets)} targets for {len(self.devices)} devices"
            )

        self.sync_replicas()

        def _one(job: tuple[int, Any]) -> dict[str, Any]:
            slot, target = job
            device = self.devices[slot]
            replica = self.replicas[slot]
            out: dict[str, Any] = {
                "target": target,
                "device": str(device),
                "oracle_calls": 0,
                "error": None,
            }
            try:
                condition = self.condition_factory(
                    target,
                    condition_noise_scale=self.condition_noise_scale,
                    device=device,
                    round_id=outer_step + slot,
                )
                state = rollout_state_factory(outer_step + slot, device)
                with torch.no_grad():
                    rollout_batch = self.rollout.generate(target, condition, replica, state)
                # Oracle scoring is subprocess-bound and thread-safe via the pool,
                # so several targets can fold concurrently.
                scored_batch = self.rewarder.score(rollout_batch, target)
                out.update(
                    condition=condition,
                    rollout_batch=rollout_batch,
                    scored_batch=scored_batch,
                    oracle_calls=int(getattr(scored_batch, "oracle_calls", 0)),
                )
            except Exception as exc:  # pragma: no cover - defensive
                out["error"] = repr(exc)
            return out

        jobs = list(enumerate(targets))
        return list(self._executor.map(_one, jobs))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _module_device(module: Any) -> torch.device | None:
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError):
        return None
