"""Thin RIDER RL engine orchestration."""

from __future__ import annotations

import time

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .checkpoint import CheckpointCursor, save_checkpoint
from .evaluation import summarize_evaluation_results, summarize_target_evaluation
from .metrics_verl import build_train_metrics
from .rollout import RolloutState
from .targets import materialize_condition


@dataclass
class RuntimeBudget:
    max_outer_steps: int = 1
    max_oracle_calls: int = 64
    max_wall_hours: float = 1.0
    checkpoint_every_oracle_calls: int = 32


@dataclass
class EngineState:
    outer_step: int = 0
    target_cursor: int = 0
    oracle_calls: int = 0
    skipped_targets: int = 0
    started_at: float = field(default_factory=time.time)



def _fallback_config_path() -> str:
    """Last resort if the launcher forgot to set engine.config_path.

    Deliberately loud rather than defaulting to some other run's YAML: the
    validation workers must use this run's validation.devices, otherwise fold
    workers could land on the training GPUs.
    """
    import os

    value = os.environ.get("RIDE_RL_CONFIG")
    if value:
        return value
    raise RuntimeError(
        "engine.config_path is unset and RIDE_RL_CONFIG is empty; refusing to "
        "guess a config for the validation worker"
    )


class RLEngine:
    """Coordinates rollout, scoring/advantage, trainer, logging, and checkpoints."""

    def __init__(
        self,
        *,
        targets: Iterable[Any],
        rollout: Any,
        rewarder: Any,
        advantage: Any,
        trainer: Any,
        old_policy: Any = None,
        budget: RuntimeBudget | Mapping[str, Any] | None = None,
        logger: Any = None,
        checkpoint_dir: str | Path | None = None,
        resolved_config: Mapping[str, Any] | None = None,
        manifest: Mapping[str, Any] | None = None,
        condition_factory: Any = materialize_condition,
        condition_noise_scale: float = 0.0,
        device: str = "cpu",
        val_fn: Any = None,
        test_freq: int = 0,
        val_before_train: bool = False,
    ) -> None:
        self.targets = list(targets)
        self.rollout = rollout
        self.rewarder = rewarder
        self.advantage = advantage
        self.trainer = trainer
        self.old_policy = old_policy if old_policy is not None else getattr(trainer, "old_policy", None)
        self.budget = self._coerce_budget(budget)
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.resolved_config = resolved_config or {}
        self.manifest = manifest or {}
        self.condition_factory = condition_factory
        self.condition_noise_scale = float(condition_noise_scale)
        self.device = device
        self.state = EngineState()
        # Periodic in-loop validation. `test_freq` counts optimizer updates, so
        # "every 10 steps" means every 10 parameter updates -- matching the
        # conventional definition of a training step.
        self.val_fn = val_fn
        self.test_freq = int(test_freq)
        self.val_before_train = bool(val_before_train)

    @staticmethod
    def _coerce_budget(budget: RuntimeBudget | Mapping[str, Any] | None) -> RuntimeBudget:
        if isinstance(budget, RuntimeBudget):
            return budget
        values = budget or {}
        return RuntimeBudget(**{k: v for k, v in values.items() if k in RuntimeBudget.__dataclass_fields__})

    def _enqueue_async_validation(self, *, step: int, tag: str) -> None:
        """Snapshot the policy and score it in a detached process.

        Keeps the training loop collective-safe: the only work done on the
        critical path is a state_dict dump. The spawned worker is fully
        detached (new session, no inherited torchrun env) so it cannot join or
        disturb the training process group.
        """
        if self.val_fn is None:
            return
        try:
            import os, subprocess, sys, json as _json
            base = self.checkpoint_dir.parent if self.checkpoint_dir is not None else Path(".")
            snap_dir = base / "validation"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap = snap_dir / f"policy_step{step:06d}.pt"
            torch.save({"model": self.trainer.current_policy.state_dict()}, snap)

            env = dict(os.environ)
            # Strip torchrun's rendezvous vars, or the child joins our group.
            for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR",
                      "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "LOCAL_WORLD_SIZE",
                      "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID"):
                env.pop(k, None)
            # MUST pass this run's --config: async_validate.py reads its
            # validation.devices, and reading another run's YAML can put fold
            # workers on the cards this run's training ranks are using. rank0
            # then queues behind them at its next CUDA sync while the other
            # ranks race ahead into the next all_reduce.
            cmd = [sys.executable, "-u", "async_validate.py",
                   "--snapshot", str(snap), "--step", str(step), "--tag", tag,
                   "--out", str(snap_dir),
                   "--config", str(getattr(self, "config_path", None) or _fallback_config_path())]
            subprocess.Popen(cmd, env=env, start_new_session=True,
                             stdout=open(snap_dir / f"eval_step{step:06d}.log", "w"),
                             stderr=subprocess.STDOUT, cwd=str(Path.cwd()))
            print(f"[validation] step={step} dispatched background evaluation -> {snap.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[validation] step={step} dispatch failed: {exc!r}", flush=True)

    def _run_validation(self, *, step: int, tag: str) -> dict[str, Any] | None:
        """Evaluate the current policy on the held-out set, mid-training.

        Runs on the main rank only; the other ranks wait at the barrier in the
        caller. Failures are logged and swallowed -- a broken validation must not
        take down a training run that is otherwise healthy.
        """
        if self.val_fn is None:
            return None
        started = time.perf_counter()
        try:
            summary = self.val_fn(step=step, tag=tag, engine=self)
        except Exception as exc:  # noqa: BLE001 - validation is best-effort
            print(f"[validation] step={step} failed: {exc!r}", flush=True)
            if self.logger is not None:
                self.logger.log("validation_error", {"step": step, "error": repr(exc)}, step=step)
            return None
        elapsed = time.perf_counter() - started
        if self.logger is not None:
            payload = {"step": step, "tag": tag, "elapsed_sec": elapsed, **(summary or {})}
            self.logger.log("validation", payload, step=step)
            if hasattr(self.logger, "log_metrics") and summary:
                metrics = {f"val/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}
                metrics["val/elapsed_sec"] = elapsed
                self.logger.log_metrics(metrics, step=step)
        print(f"[validation] step={step} tag={tag} took {elapsed:.1f}s", flush=True)
        return summary

    def _run_ddp(self) -> None:
        """Synchronous data-parallel loop; one optimizer update per step.

            1 step == 1 batch == 1 optimizer update

        The batch spans `world_size / ranks_per_target` targets. Each rank rolls
        out its slice of one target's group on its own GPU, folds that slice with
        its own oracle worker, and backwards its own loss; gradients are averaged
        across all ranks inside `train_step`, so every GPU does real work in all
        three phases. The phases run strictly sequentially.

        Each rank's loss is the mean over its own samples, so averaging across
        ranks gives `(1/T) sum_targets (1/G) sum_samples` -- exactly the intended
        batch objective, with no extra weighting needed.
        """
        from . import distributed as dist_utils

        world = dist_utils.world_size()
        rank = dist_utils.rank()
        is_main = dist_utils.is_main()
        runtime_cfg = dict(self.resolved_config.get("runtime", {}))
        base_seed = int(runtime_cfg.get("seed", 0))

        rpt = max(1, int(runtime_cfg.get("ranks_per_target", 1)))
        if world % rpt != 0:
            raise ValueError(f"world_size {world} not divisible by ranks_per_target {rpt}")
        dist_utils.init_target_group(rpt)

        batch_targets = world // rpt
        slot = dist_utils.target_slot()
        my_target_index = dist_utils.target_id()

        group_size = int(self.resolved_config.get("rollout", {}).get("group_size", 8))
        if group_size % rpt != 0:
            raise ValueError(f"group_size {group_size} not divisible by ranks_per_target {rpt}")
        per_rank = group_size // rpt
        my_indices = tuple(range(slot * per_rank, (slot + 1) * per_rank))

        # Replicas start bit-identical so the on-policy assumption holds.
        dist_utils.broadcast_module(self.trainer.current_policy)
        if self.old_policy is not None:
            dist_utils.broadcast_module(self.old_policy)
        if getattr(self.trainer, "reference_policy", None) is not None:
            dist_utils.broadcast_module(self.trainer.reference_policy)

        total_steps = len(self.targets) // batch_targets
        max_steps = min(total_steps, int(self.budget.max_outer_steps))

        if is_main:
            print(
                f"[ddp] world={world} ranks_per_target={rpt} batch_targets={batch_targets} "
                f"group_size={group_size} per_rank={per_rank} "
                f"samples_per_update={batch_targets * group_size} total_steps={max_steps}",
                flush=True,
            )

        # Must NOT run inline: rank0 folding RNA for minutes while the other
        # ranks wait in the barrier below is exactly the rank-divergence that
        # hung the earlier run. Dispatch it out-of-process like the periodic
        # path, so every rank reaches the barrier in the same instant.
        if self.val_fn is not None and self.val_before_train and is_main:
            self._enqueue_async_validation(step=0, tag="before_train")
        # This barrier is safe: it is outside the step loop and every rank
        # reaches it unconditionally, with no other collective in flight.
        dist_utils.barrier()

        try:
            for step in range(max_steps):
                if self._budget_exhausted_ddp(step, max_steps):
                    break
                step_started = time.perf_counter()
                cursor = step * batch_targets + my_target_index
                target = self.targets[cursor]

                phase: dict[str, float] = {}
                t0 = time.perf_counter()
                condition = self.condition_factory(
                    target,
                    condition_noise_scale=self.condition_noise_scale,
                    device=self.device,
                    round_id=step,
                )
                phase["condition"] = time.perf_counter() - t0

                rollout_state = RolloutState(
                    round_id=step,
                    step=step,
                    # Seed depends on the target, not the rank, so ranks sharing
                    # a target produce complementary halves of the same group.
                    base_seed=base_seed + step * 100003 + my_target_index * 9973,
                    policy_version=f"old@{getattr(self.trainer.state, 'optimizer_step', 0)}",
                )

                t0 = time.perf_counter()
                with torch.no_grad():
                    rollout_batch = self.rollout.generate(
                        target, condition, self.old_policy, rollout_state,
                        sample_indices=my_indices,
                    )
                phase["generate"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                scored_batch = self.rewarder.score(rollout_batch, target)
                phase["score"] = time.perf_counter() - t0

                oracle_calls = int(getattr(scored_batch, "oracle_calls", 0))
                self.state.oracle_calls += oracle_calls

                # GRPO needs statistics over the target's *whole* group; merge
                # the sharing ranks' rewards before normalising.
                t0 = time.perf_counter()
                group_stats = None
                if rpt > 1:
                    records = getattr(getattr(scored_batch, "rewards", None), "records", ())
                    local_rewards = [
                        float(r.raw_reward)
                        for r in records
                        if getattr(r, "raw_reward", None) is not None
                    ]
                    mean, std, n = dist_utils.merged_group_stats(local_rewards)
                    if n >= 2 and std > 0:
                        group_stats = (mean, std)
                train_batch = self.advantage.compute(scored_batch, group_stats=group_stats)
                phase["advantage"] = time.perf_counter() - t0

                # A rank with nothing to train on must still join the collective,
                # contributing zero gradients; skipping would deadlock the rest.
                t0 = time.perf_counter()
                if train_batch is None or bool(getattr(train_batch, "skip_update", False)):
                    self.state.skipped_targets += 1
                    train_result = self.trainer.train_step_empty()
                    status = "skipped"
                else:
                    train_result = self.trainer.train_step(train_batch)
                    status = train_result.get("status", "ok")
                phase["train_step"] = time.perf_counter() - t0
                phase["total"] = time.perf_counter() - step_started

                step_result = {
                    "status": status,
                    "target_id": getattr(target, "target_id", None),
                    "oracle_calls": oracle_calls,
                    "reward": getattr(scored_batch, "cache_summary", {}),
                    "paper_metrics": summarize_target_evaluation(
                        target=target,
                        rollout_samples=getattr(rollout_batch, "samples", ()),
                        reward_summary=getattr(scored_batch, "cache_summary", {}),
                        reward_records=getattr(getattr(scored_batch, "rewards", None), "records", ()),
                        oracle_calls=oracle_calls,
                    ),
                    "rollout": getattr(self.rollout, "state_dict", lambda: {})(),
                    "trainer": train_result,
                    "gpu_peak_mb": _gpu_peak_mb(),
                    "timing": phase,
                    "rank": rank,
                    "target_index": my_target_index,
                    "sample_slice": list(my_indices),
                    "device": str(self.device),
                }

                # Aggregate so the logged metrics describe the whole batch.
                local = {
                    "reward_mean": float((step_result["reward"] or {}).get("reward_mean") or 0.0),
                    "good_rate": float((step_result["paper_metrics"] or {}).get("good_rate") or 0.0),
                    "recovery": float((step_result["paper_metrics"] or {}).get("sequence_recovery") or 0.0),
                    "oracle_calls": float(oracle_calls),
                }
                gathered = dist_utils.gather_object(local)

                if is_main:
                    self.state.outer_step = step + 1
                    self.state.target_cursor = (step + 1) * batch_targets
                    if self.logger is not None:
                        self.logger.log("outer_step", step_result, step=step)
                        if hasattr(self.logger, "log_metrics"):
                            n = max(1, len(gathered))
                            metrics = {
                                "batch/reward_mean": sum(g["reward_mean"] for g in gathered) / n,
                                "batch/good_rate": sum(g["good_rate"] for g in gathered) / n,
                                "batch/recovery": sum(g["recovery"] for g in gathered) / n,
                                "batch/oracle_calls": sum(g["oracle_calls"] for g in gathered),
                                "batch/size_targets": float(batch_targets),
                                "batch/size_samples": float(batch_targets * group_size),
                                "batch/ranks_per_target": float(rpt),
                                "train/step": float(step),
                                "train/epoch": float((step + 1) * batch_targets) / float(max(1, len(self.targets))),
                                "time/step_sec": phase["total"],
                                "time/generate_sec": phase["generate"],
                                "time/score_sec": phase["score"],
                                "time/train_sec": phase["train_step"],
                            }
                            for key in ("loss", "grad_norm", "lr", "kl", "nft_loss", "reference_loss"):
                                value = train_result.get(key)
                                if isinstance(value, (int, float)):
                                    metrics[f"train/{key}"] = float(value)
                            self.logger.log_metrics(metrics, step=step)

                # test_freq counts optimizer updates, per the batch definition.
                if (
                    self.val_fn is not None
                    and self.test_freq > 0
                    and (step + 1) % self.test_freq == 0
                ):
                    # Validation used to run inline on rank0 while ranks 1-3 sat in
                    # the barrier below. That stalls one rank for ~270s every 10
                    # steps, and the process group does not survive it: rank0 was
                    # repeatedly found still inside this barrier while the others
                    # had already moved on to the next step's all-reduce. Instead
                    # rank0 now only dumps a policy snapshot (fast, no GPU work,
                    # no subprocess) and a detached worker scores it out-of-band,
                    # so no rank is ever blocked for more than a moment.
                    # No barrier here. Dispatch is a local, non-collective act
                    # (dump a state_dict, spawn a detached process); the
                    # gradient all_reduce every rank runs inside train_step is
                    # already the step's synchronisation point. Keeping an extra
                    # barrier gave NCCL a second collective to mismatch: rank0
                    # was caught in barrier() while ranks 1-3 were in
                    # average_gradients(), and the two paired up and hung.
                    if is_main:
                        self._enqueue_async_validation(step=step + 1, tag="periodic")

                if is_main and self._should_checkpoint():
                    self.save_checkpoint("latest.pt")
        finally:
            if hasattr(self.rewarder, "close"):
                self.rewarder.close()

        dist_utils.barrier()
        summary = {
            "status": "complete",
            "outer_step": self.state.outer_step,
            "target_cursor": self.state.target_cursor,
            "oracle_calls": self.state.oracle_calls,
            "skipped_targets": self.state.skipped_targets,
            "world_size": world,
            "ranks_per_target": rpt,
            "batch_targets": batch_targets,
        }
        if is_main and self.checkpoint_dir is not None and self._checkpoint_boundary():
            self.save_checkpoint("final.pt")
        return summary

    def _budget_exhausted_ddp(self, step: int, max_steps: int) -> bool:
        if step >= max_steps:
            return True
        if self.state.oracle_calls >= self.budget.max_oracle_calls:
            return True
        elapsed_hours = (time.time() - self.state.started_at) / 3600.0
        return elapsed_hours >= self.budget.max_wall_hours

    def run(self) -> dict[str, Any]:
        from . import distributed as _du

        if _du.is_enabled():
            return self._run_ddp()
        try:
            while not self._budget_exhausted():
                if self.state.target_cursor >= len(self.targets):
                    break
                target = self.targets[self.state.target_cursor]
                step_result = self.run_one_target(target)
                if self.logger is not None:
                    self.logger.log("outer_step", step_result, step=self.state.outer_step)
                self.state.outer_step += 1
                self.state.target_cursor += 1
                if self._should_checkpoint():
                    self.save_checkpoint("latest.pt")
        finally:
            if hasattr(self.rewarder, "close"):
                self.rewarder.close()
        summary = {
            "status": "complete",
            "outer_step": self.state.outer_step,
            "target_cursor": self.state.target_cursor,
            "oracle_calls": self.state.oracle_calls,
            "skipped_targets": self.state.skipped_targets,
        }
        if self.checkpoint_dir is not None and self._checkpoint_boundary():
            self.save_checkpoint("final.pt")
        return summary

    def evaluate(self) -> dict[str, Any]:
        """Run rollout + reward only; never mutate trainer, optimizer, or EMA."""
        results: list[dict[str, Any]] = []
        try:
            while not self._budget_exhausted():
                if self.state.target_cursor >= len(self.targets):
                    break
                target = self.targets[self.state.target_cursor]
                result = self.evaluate_one_target(target)
                if self.logger is not None:
                    self.logger.log("eval_step", result, step=self.state.outer_step)
                results.append(result)
                self.state.outer_step += 1
                self.state.target_cursor += 1
        finally:
            if hasattr(self.rewarder, "close"):
                self.rewarder.close()
        return summarize_evaluation_results(
            results,
            outer_step=self.state.outer_step,
            target_cursor=self.state.target_cursor,
            oracle_calls=self.state.oracle_calls,
            skipped_targets=self.state.skipped_targets,
        )

    def run_one_target(self, target: Any) -> dict[str, Any]:
        # All formulas live in the collaborators. Engine only forwards typed batches.
        condition = self.condition_factory(
            target,
            condition_noise_scale=self.condition_noise_scale,
            device=self.device,
            round_id=self.state.outer_step,
        )
        rollout_state = RolloutState(
            round_id=self.state.outer_step,
            step=self.state.outer_step,
            base_seed=int(self.resolved_config.get("runtime", {}).get("seed", 0)) + self.state.outer_step * 100000,
            policy_version=f"old@{getattr(self.trainer.state, 'optimizer_step', 0)}",
        )
        rollout_batch, scored_batch, rollout_state, oracle_calls = self._rollout_and_score_with_rescue(target, condition, rollout_state)
        self.state.oracle_calls += oracle_calls
        train_batch = self.advantage.compute(scored_batch)
        if train_batch is None or bool(getattr(train_batch, "skip_update", False)):
            self.state.skipped_targets += 1
            return {
                "status": "skipped",
                "target_id": getattr(target, "target_id", None),
                "oracle_calls": oracle_calls,
                "skip_reason": getattr(train_batch, "skip_reason", "no_train_batch") if train_batch is not None else "no_train_batch",
                "reward": getattr(scored_batch, "cache_summary", {}),
                "paper_metrics": summarize_target_evaluation(
                    target=target,
                    rollout_samples=getattr(rollout_batch, "samples", ()),
                    reward_summary=getattr(scored_batch, "cache_summary", {}),
                    reward_records=getattr(getattr(scored_batch, "rewards", None), "records", ()),
                    oracle_calls=oracle_calls,
                ),
                "rollout": getattr(self.rollout, "state_dict", lambda: {})(),
                "gpu_peak_mb": _gpu_peak_mb(),
            }
        train_result = self.trainer.train_step(train_batch)
        return {
            "status": train_result.get("status", "ok"),
            "target_id": getattr(target, "target_id", None),
            "oracle_calls": oracle_calls,
            "reward": getattr(scored_batch, "cache_summary", {}),
            "paper_metrics": summarize_target_evaluation(
                target=target,
                rollout_samples=getattr(rollout_batch, "samples", ()),
                reward_summary=getattr(scored_batch, "cache_summary", {}),
                reward_records=getattr(getattr(scored_batch, "rewards", None), "records", ()),
                oracle_calls=oracle_calls,
            ),
            "rollout": getattr(self.rollout, "state_dict", lambda: {})(),
            "trainer": train_result,
            "gpu_peak_mb": _gpu_peak_mb(),
        }

    def evaluate_one_target(self, target: Any) -> dict[str, Any]:
        condition = self.condition_factory(
            target,
            condition_noise_scale=self.condition_noise_scale,
            device=self.device,
            round_id=self.state.outer_step,
        )
        rollout_state = RolloutState(
            round_id=self.state.outer_step,
            step=self.state.outer_step,
            base_seed=int(self.resolved_config.get("runtime", {}).get("seed", 0)) + self.state.outer_step * 100000,
            policy_version=f"eval@{getattr(self.trainer.state, 'optimizer_step', 0)}",
        )
        rollout_batch, scored_batch, _, oracle_calls = self._rollout_and_score_with_rescue(target, condition, rollout_state)
        self.state.oracle_calls += oracle_calls
        return {
            "status": "ok",
            "target_id": getattr(target, "target_id", None),
            "oracle_calls": oracle_calls,
            "reward": getattr(scored_batch, "cache_summary", {}),
            "paper_metrics": summarize_target_evaluation(
                target=target,
                rollout_samples=getattr(rollout_batch, "samples", ()),
                reward_summary=getattr(scored_batch, "cache_summary", {}),
                reward_records=getattr(getattr(scored_batch, "rewards", None), "records", ()),
                oracle_calls=oracle_calls,
            ),
            "rollout": getattr(self.rollout, "state_dict", lambda: {})(),
            "gpu_peak_mb": _gpu_peak_mb(),
        }

    def _rollout_and_score_with_rescue(self, target: Any, condition: Any, rollout_state: RolloutState) -> tuple[Any, Any, RolloutState, int]:
        total_oracle_calls = 0
        state = rollout_state
        while True:
            rollout_batch = self.rollout.generate(target, condition, self.old_policy, state)
            scored_batch = self.rewarder.score(rollout_batch, target)
            total_oracle_calls += int(getattr(scored_batch, "oracle_calls", 0))
            next_state = None
            if hasattr(self.rollout, "next_round"):
                next_state = self.rollout.next_round(scored_batch, state)
            if next_state is None:
                return rollout_batch, scored_batch, state, total_oracle_calls
            state = next_state

    def _budget_exhausted(self) -> bool:
        if self.state.outer_step >= self.budget.max_outer_steps:
            return True
        if self.state.oracle_calls >= self.budget.max_oracle_calls:
            return True
        elapsed_hours = (time.time() - self.state.started_at) / 3600.0
        return elapsed_hours >= self.budget.max_wall_hours

    def _should_checkpoint(self) -> bool:
        if self.checkpoint_dir is None:
            return False
        every = max(1, int(self.budget.checkpoint_every_oracle_calls))
        return self.state.oracle_calls > 0 and self.state.oracle_calls % every == 0 and self._checkpoint_boundary()

    def _checkpoint_boundary(self) -> bool:
        checker = getattr(self.trainer, "can_checkpoint", None)
        return bool(checker()) if checker is not None else True

    def save_checkpoint(self, name: str) -> Path:
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not configured")
        return save_checkpoint(
            self.checkpoint_dir / name,
            current_policy=self.trainer.current_policy,
            old_policy=self.trainer.old_policy,
            reference_policy=self.trainer.reference_policy,
            optimizer=self.trainer.optimizer,
            amp_scaler=getattr(self.trainer, "scaler", None),
            cursor=CheckpointCursor(
                outer_step=self.state.outer_step,
                optimizer_step=self.trainer.state.optimizer_step,
                target_cursor=self.state.target_cursor,
                oracle_calls=self.state.oracle_calls,
            ),
            resolved_config=self.resolved_config,
            reward_scale_state=getattr(self.advantage, "state_dict", lambda: {})(),
            temperature_state=getattr(self.rollout, "state_dict", lambda: {})(),
            manifest=self.manifest,
            extra_state={"engine": self.state.__dict__, "trainer": self.trainer.state_dict()},
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        engine_state = state.get("engine", state)
        for key, value in engine_state.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        if "trainer" in state and hasattr(self.trainer, "load_state_dict"):
            self.trainer.load_state_dict(state["trainer"])
        if "reward_scale_state" in state and hasattr(self.advantage, "load_state_dict"):
            self.advantage.load_state_dict(state["reward_scale_state"])
        if "temperature_state" in state and hasattr(self.rollout, "load_state_dict"):
            self.rollout.load_state_dict(state["temperature_state"])


def _gpu_peak_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        return None
