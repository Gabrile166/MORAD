"""Evaluation helpers for the RiboDiffusion-style benchmark summary."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src.rl.reward import unique_sequences


def length_bucket(length: int) -> str:
    """Match the paper's short / medium / long reporting buckets."""

    if length <= 50:
        return "short"
    if length <= 100:
        return "medium"
    return "long"


def target_type(target: Any) -> str | None:
    """Best-effort RNA family/type label when the dataset has one."""

    raw = getattr(target, "raw_record", None)
    if not isinstance(raw, Mapping):
        return None
    for key in ("rna_type", "type", "family", "rna_family"):
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def sequence_recovery(sample_tokens: str, native_sequence: str) -> float:
    if not sample_tokens or not native_sequence:
        return 0.0
    length = min(len(sample_tokens), len(native_sequence))
    if length <= 0:
        return 0.0
    matches = sum(1 for lhs, rhs in zip(sample_tokens[:length], native_sequence[:length]) if lhs == rhs)
    return matches / float(length)


def summarize_target_evaluation(
    *,
    target: Any,
    rollout_samples: Sequence[Any],
    reward_summary: Mapping[str, Any],
    reward_records: Sequence[Any],
    oracle_calls: int,
) -> dict[str, Any]:
    native = str(getattr(target, "sequence_native", ""))
    lengths = int(getattr(target, "length", len(native)))
    sequences = [str(getattr(sample, "tokens", "")) for sample in rollout_samples]
    recoveries = [sequence_recovery(sequence, native) for sequence in sequences]
    unique = unique_sequences(sequences)
    valid_records = [record for record in reward_records if getattr(record, "status", None) == "ok"]
    good_records = [record for record in valid_records if bool(getattr(record, "is_good", False))]
    exact_matches = sum(1 for sequence in sequences if sequence == native)

    summary = {
        "target_id": getattr(target, "target_id", None),
        "split": getattr(target, "split", None),
        "length": lengths,
        "length_bucket": length_bucket(lengths),
        "rna_type": target_type(target),
        "sample_count": len(sequences),
        "unique_sequence_count": len(unique),
        "unique_sequence_ratio": (len(unique) / len(sequences)) if sequences else 0.0,
        "duplicate_ratio": (1.0 - len(unique) / len(sequences)) if sequences else 0.0,
        "sequence_recovery_mean": mean(recoveries) if recoveries else 0.0,
        "sequence_recovery_best": max(recoveries) if recoveries else 0.0,
        "sequence_exact_match_rate": (exact_matches / len(sequences)) if sequences else 0.0,
        "valid_sample_count": len(valid_records),
        "valid_rate": (len(valid_records) / len(sequences)) if sequences else 0.0,
        "good_sample_count": len(good_records),
        "good_rate": (len(good_records) / len(sequences)) if sequences else 0.0,
        "oracle_calls": int(oracle_calls),
        "reward": dict(reward_summary),
    }
    summary["tm_score_mean"] = float(summary["reward"].get("tm_score_mean", 0.0) or 0.0)
    summary["rmsd_mean"] = float(summary["reward"].get("rmsd_mean", 0.0) or 0.0)
    summary["gdt_ts_mean"] = float(summary["reward"].get("gdt_ts_mean", 0.0) or 0.0)
    summary["fold_cache_hits"] = int(summary["reward"].get("fold_cache_hits", 0) or 0)
    summary["fold_cache_misses"] = int(summary["reward"].get("fold_cache_misses", 0) or 0)
    summary["metric_cache_hits"] = int(summary["reward"].get("metric_cache_hits", 0) or 0)
    summary["metric_cache_misses"] = int(summary["reward"].get("metric_cache_misses", 0) or 0)
    summary["oracle_cache_hit_rate"] = (
        summary["fold_cache_hits"] / max(1, summary["fold_cache_hits"] + summary["fold_cache_misses"])
    )
    return summary


def summarize_evaluation_results(
    results: Sequence[Mapping[str, Any]],
    *,
    outer_step: int,
    target_cursor: int,
    oracle_calls: int,
    skipped_targets: int,
) -> dict[str, Any]:
    if not results:
        return {
            "status": "complete",
            "mode": "eval",
            "outer_step": int(outer_step),
            "target_cursor": int(target_cursor),
            "oracle_calls": int(oracle_calls),
            "skipped_targets": int(skipped_targets),
            "targets": 0,
            "paper_summary": {},
            "results": [],
        }

    def _mean_field(items: Iterable[Mapping[str, Any]], key: str) -> float:
        values = [float(_metric_view(item).get(key)) for item in items if _metric_view(item).get(key) is not None]
        return mean(values) if values else 0.0

    def _metric_view(item: Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(item.get("paper_metrics"), Mapping):
            return item["paper_metrics"]  # type: ignore[return-value]
        return item

    paper_summary: dict[str, Any] = {
        "targets": len(results),
        "sequence_recovery_mean": _mean_field(results, "sequence_recovery_mean"),
        "sequence_recovery_best_mean": _mean_field(results, "sequence_recovery_best"),
        "sequence_exact_match_rate_mean": _mean_field(results, "sequence_exact_match_rate"),
        "unique_sequence_ratio_mean": _mean_field(results, "unique_sequence_ratio"),
        "duplicate_ratio_mean": _mean_field(results, "duplicate_ratio"),
        "good_rate_mean": _mean_field(results, "good_rate"),
        "valid_rate_mean": _mean_field(results, "valid_rate"),
        "rmsd_mean": _mean_field(results, "rmsd_mean"),
        "tm_score_mean": _mean_field(results, "tm_score_mean"),
        "gdt_ts_mean": _mean_field(results, "gdt_ts_mean"),
        "oracle_cache_hit_rate_mean": _mean_field(results, "oracle_cache_hit_rate"),
    }

    length_buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    type_buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in results:
        view = _metric_view(item)
        length_buckets[str(view.get("length_bucket", "unknown"))].append(item)
        rna_type = view.get("rna_type")
        if rna_type not in (None, ""):
            type_buckets[str(rna_type)].append(item)

    if length_buckets:
        paper_summary["length_buckets"] = {
            bucket: {
                "count": len(items),
                "sequence_recovery_mean": _mean_field(items, "sequence_recovery_mean"),
                "unique_sequence_ratio_mean": _mean_field(items, "unique_sequence_ratio"),
                "good_rate_mean": _mean_field(items, "good_rate"),
                "rmsd_mean": _mean_field(items, "rmsd_mean"),
                "tm_score_mean": _mean_field(items, "tm_score_mean"),
                "gdt_ts_mean": _mean_field(items, "gdt_ts_mean"),
            }
            for bucket, items in sorted(length_buckets.items())
        }
    if type_buckets:
        paper_summary["rna_type_buckets"] = {
            rna_type: {
                "count": len(items),
                "sequence_recovery_mean": _mean_field(items, "sequence_recovery_mean"),
                "unique_sequence_ratio_mean": _mean_field(items, "unique_sequence_ratio"),
                "good_rate_mean": _mean_field(items, "good_rate"),
                "rmsd_mean": _mean_field(items, "rmsd_mean"),
                "tm_score_mean": _mean_field(items, "tm_score_mean"),
                "gdt_ts_mean": _mean_field(items, "gdt_ts_mean"),
            }
            for rna_type, items in sorted(type_buckets.items())
        }

    return {
        "status": "complete",
        "mode": "eval",
        "outer_step": int(outer_step),
        "target_cursor": int(target_cursor),
        "oracle_calls": int(oracle_calls),
        "skipped_targets": int(skipped_targets),
        "targets": len(results),
        "paper_summary": paper_summary,
        "results": list(results),
    }
