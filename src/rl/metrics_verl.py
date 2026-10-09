"""verl-style metric aggregation and W&B tracking for MORAD.

Mirrors the metric conventions used by verl's RayPPOTrainer:

* namespaced keys -- ``reward/*``, ``actor/*``, ``rollout/*``, ``timing/*``,
  ``val/*`` -- so the W&B UI groups related panels together;
* mean/max/min for every distribution-valued quantity;
* a single ``logger.log(data=metrics, step=global_step)`` per outer step, with
  periodic validation metrics merged into that same call.

Reference: verl/trainer/ppo/ray_trainer.py :: fit() and metric_utils.py.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Any, Iterable, Mapping, MutableMapping


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def dist_metrics(prefix: str, values: Iterable[Any], with_std: bool = True) -> dict[str, float]:
    """mean/max/min(/std) for one distribution, verl-style."""
    vals = _finite(values)
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    out = {
        f"{prefix}/mean": mean,
        f"{prefix}/max": max(vals),
        f"{prefix}/min": min(vals),
    }
    if with_std and n > 1:
        var = sum((v - mean) ** 2 for v in vals) / n
        out[f"{prefix}/std"] = math.sqrt(var)
    return out


@contextmanager
def timer(metrics: MutableMapping[str, Any], key: str):
    """Record wall time into ``metrics[f'timing/{key}']`` like verl's Timer."""
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[f"timing/{key}"] = time.perf_counter() - start


def build_train_metrics(step_result: Mapping[str, Any]) -> dict[str, float]:
    """Flatten one outer-step result into verl-style namespaced metrics.

    Covers the quantities a verl user expects to see on the dashboard:
    reward and every reward sub-term, advantage stats, loss decomposition
    (policy / positive / negative / reference-KL), grad norm, lr, rollout
    diversity and oracle/cache behaviour.
    """
    m: dict[str, float] = {}

    reward = step_result.get("reward") or {}
    if isinstance(reward, Mapping):
        # Scalar reward and its structural sub-terms.
        for src, dst in (
            ("reward_mean", "reward/score/mean"),
            ("reward_std", "reward/score/std"),
            ("gdt_ts_mean", "reward/gdt_ts/mean"),
            ("tm_score_mean", "reward/tm_score/mean"),
            ("rmsd_mean", "reward/rmsd/mean"),
        ):
            v = reward.get(src)
            if v is not None:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f):
                    m[dst] = f
        for src, dst in (
            ("invalid_records", "reward/invalid_records"),
            ("unique_sequences", "rollout/unique_sequences"),
            ("fold_cache_hits", "oracle/fold_cache_hits"),
            ("fold_cache_misses", "oracle/fold_cache_misses"),
            ("metric_cache_hits", "oracle/metric_cache_hits"),
            ("metric_cache_misses", "oracle/metric_cache_misses"),
        ):
            v = reward.get(src)
            if v is not None:
                m[dst] = float(v)

    # Per-sample reward distribution and advantages, when the engine exposes them.
    for key, prefix in (
        ("reward_values", "reward/score"),
        ("advantages", "actor/advantage"),
        ("optimality", "actor/optimality"),
    ):
        seq = step_result.get(key)
        if isinstance(seq, (list, tuple)) and seq:
            m.update(dist_metrics(prefix, seq))

    trainer = step_result.get("trainer") or {}
    if isinstance(trainer, Mapping):
        for src, dst in (
            ("loss_total", "actor/loss"),
            ("loss_policy", "actor/loss_policy"),
            ("loss_positive", "actor/loss_positive"),
            ("loss_negative", "actor/loss_negative"),
            ("loss_reference", "actor/kl_loss"),
            ("grad_norm", "actor/grad_norm"),
            ("lr", "actor/lr"),
            ("ema_decay", "actor/ema_decay"),
            ("optimizer_step", "actor/optimizer_step"),
        ):
            v = trainer.get(src)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                m[dst] = f
        status = trainer.get("status")
        if status is not None:
            m["actor/update_ok"] = 1.0 if status == "ok" else 0.0

    paper = step_result.get("paper_metrics") or {}
    if isinstance(paper, Mapping):
        for src, dst in (
            ("good_rate", "reward/good_rate"),
            ("good_sample_count", "reward/good_count"),
            ("duplicate_ratio", "rollout/duplicate_ratio"),
            ("length", "rollout/target_length"),
            ("oracle_calls", "oracle/calls_this_step"),
            ("oracle_cache_hit_rate", "oracle/cache_hit_rate"),
        ):
            v = paper.get(src)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                m[dst] = f

    rollout = step_result.get("rollout") or {}
    if isinstance(rollout, Mapping):
        for src, dst in (
            ("last_temperature", "rollout/temperature"),
            ("rescue_count", "rollout/rescue_count"),
        ):
            v = rollout.get(src)
            if v is None:
                continue
            try:
                m[dst] = float(v)
            except (TypeError, ValueError):
                pass

    v = step_result.get("oracle_calls")
    if v is not None:
        try:
            m["oracle/calls_cumulative"] = float(v)
        except (TypeError, ValueError):
            pass

    status = step_result.get("status")
    if status is not None:
        m["train/step_ok"] = 1.0 if status == "ok" else 0.0
    return m


def build_val_metrics(summary: Mapping[str, Any], prefix: str = "val") -> dict[str, float]:
    """Flatten an evaluate.py summary.json into ``val/*`` metrics.

    verl merges validation metrics into the same log call as the training
    metrics for that step, so the dashboard shows train and val on one x-axis.
    """
    m: dict[str, float] = {}

    def take(group: Mapping[str, Any] | None, names: Mapping[str, str]) -> None:
        if not isinstance(group, Mapping):
            return
        for src, dst in names.items():
            entry = group.get(src)
            if not isinstance(entry, Mapping):
                continue
            for stat in ("mean", "std", "max", "min"):
                v = entry.get(stat)
                if v is None:
                    continue
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f):
                    m[f"{prefix}/{dst}/{stat}"] = f
            ok = entry.get("ok")
            if ok is not None:
                try:
                    m[f"{prefix}/{dst}/ok"] = float(ok)
                except (TypeError, ValueError):
                    pass

    take(
        summary.get("candidate_metrics"),
        {
            "sequence_recovery": "recovery",
            "secondary_structure_f1": "rnafold_f1",
            "rfam_family_success": "rfam_success",
            "rhofold_c1prime_tm_score": "rhofold_tm",
            "reward_c4p_tm_score": "c4p_tm",
            "reward_c4p_gdt_ts": "c4p_gdt_ts",
            "reward_c4p_rmsd": "c4p_rmsd",
            "reward_good": "good_rate",
            "reward_raw_score": "reward_score",
        },
    )
    take(summary.get("target_metrics"), {"internal_diversity": "intdiv"})

    for key in ("target_count", "candidate_count"):
        v = summary.get(key)
        if v is not None:
            try:
                m[f"{prefix}/{key}"] = float(v)
            except (TypeError, ValueError):
                pass

    buckets = summary.get("by_length_bucket")
    if isinstance(buckets, Mapping):
        for bucket in ("short", "medium", "long"):
            grp = buckets.get(bucket)
            if not isinstance(grp, Mapping):
                continue
            for src, dst in (
                ("rhofold_c1prime_tm_score", "rhofold_tm"),
                ("sequence_recovery", "recovery"),
            ):
                entry = grp.get(src)
                if isinstance(entry, Mapping) and entry.get("mean") is not None:
                    try:
                        f = float(entry["mean"])
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(f):
                        m[f"{prefix}/by_bucket/{bucket}/{dst}"] = f
    return m
