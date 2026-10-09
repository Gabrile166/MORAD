"""Offline curriculum selection helpers for versioned RIDE target pools."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from src.evaluation.difficulty import DifficultyConfig, classify_difficulty
from src.evaluation.metrics import length_bucket
from src.rl.targets import SCHEMA_VERSION


@dataclass(frozen=True)
class RecoveryScreenConfig:
    limit: int = 900
    short_fraction: float = 0.30
    medium_fraction: float = 0.45
    long_fraction: float = 0.25


@dataclass(frozen=True)
class StructuralSelectionConfig:
    limit: int = 500
    min_structure_score: float = 0.05
    easy_anchor_fraction: float = 0.15


@dataclass(frozen=True)
class RNA3DBSelectionConfig:
    limit: int = 100
    recent_cutoff: str = "2024-01-01"
    recent_minimum_fraction: float = 0.20
    max_resolution: float = 3.0
    max_terminal_trim: int = 10


@dataclass(frozen=True)
class OracleSelectionConfig:
    minimum_targets: int = 500
    limit: int | None = None
    minimum_passed_checks: int = 2
    native_tm_threshold: float = 0.45
    native_gdt_threshold: float = 0.50
    native_rmsd_threshold: float = 2.0


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def select_low_recovery_pool(
    source_pool: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    config: RecoveryScreenConfig | None = None,
    *,
    source_label: str = "ride_train",
) -> dict[str, Any]:
    """Select a length-stratified buffer where frozen RIDE has low recovery."""

    cfg = config or RecoveryScreenConfig()
    candidates_by_target: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in evaluation.get("candidates", []):
        candidates_by_target.setdefault(str(candidate.get("target_id", "")), []).append(candidate)
    target_metrics = {
        str(target.get("target_id", "")): target.get("metrics", {})
        for target in evaluation.get("targets", [])
    }

    ranked_by_bucket: dict[str, list[tuple[tuple[float, float, str], Mapping[str, Any], dict[str, Any]]]] = {
        "short": [],
        "medium": [],
        "long": [],
    }
    for target in source_pool.get("targets", []):
        target_id = _pool_target_id(target)
        recoveries = [
            value
            for candidate in candidates_by_target.get(target_id, [])
            if (value := _metric_value(candidate.get("metrics", {}).get("sequence_recovery"))) is not None
        ]
        if not recoveries:
            continue
        recovery = mean(recoveries)
        diversity = _metric_value(target_metrics.get(target_id, {}).get("internal_diversity"))
        metadata = target.get("metadata", {})
        bucket = length_bucket(int(metadata.get("length", len(str(metadata.get("sequence", ""))))))
        screen = {
            "stage": "frozen_ride_recovery_screen",
            "source": source_label,
            "sequence_recovery_mean": recovery,
            "internal_diversity": diversity,
            "n_samples": len(recoveries),
            "length_bucket": bucket,
        }
        ranked_by_bucket[bucket].append(((recovery, -(diversity or 0.0), target_id), target, screen))

    quotas = _fractional_quotas(
        cfg.limit,
        {"short": cfg.short_fraction, "medium": cfg.medium_fraction, "long": cfg.long_fraction},
    )
    selected: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    leftovers: list[tuple[tuple[float, float, str], Mapping[str, Any], dict[str, Any]]] = []
    for bucket, ranked in ranked_by_bucket.items():
        ranked.sort(key=lambda item: item[0])
        selected.extend((target, screen) for _, target, screen in ranked[: quotas[bucket]])
        leftovers.extend(ranked[quotas[bucket] :])
    leftovers.sort(key=lambda item: item[0])
    for _, target, screen in leftovers:
        if len(selected) >= cfg.limit:
            break
        selected.append((target, screen))

    targets = [_with_curriculum_metadata(target, screen) for target, screen in selected[: cfg.limit]]
    return _derived_pool(
        source_pool,
        targets,
        selection="low_frozen_ride_recovery_buffer",
        config=asdict(cfg),
        extra_summary={
            "source": source_label,
            "length_counts": _counts(t["metadata"]["curriculum_screen"]["length_bucket"] for t in targets),
        },
    )


def select_structural_curriculum_pool(
    source_pool: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    config: StructuralSelectionConfig | None = None,
    *,
    source_label: str = "ride_train",
) -> dict[str, Any]:
    """Select oracle-reliable hard/medium targets plus a small easy anchor set."""

    cfg = config or StructuralSelectionConfig()
    report = classify_difficulty(
        evaluation,
        evaluation,
        DifficultyConfig(min_ride_score=cfg.min_structure_score, min_ribo_good_rate=0.0),
    )


    rows = {str(row["target_id"]): row for row in report.get("targets", [])}
    pool_targets = {_pool_target_id(target): target for target in source_pool.get("targets", [])}
    eligible: dict[str, list[tuple[float, str, Mapping[str, Any], Mapping[str, Any]]]] = {
        "hard": [],
        "medium": [],
        "easy": [],
    }
    rejection_counts: dict[str, int] = {}
    for target_id, target in pool_targets.items():
        row = rows.get(target_id)
        if row is None:
            _bump(rejection_counts, "missing_evaluation")
            continue
        if not row.get("oracle", {}).get("reliable"):
            _bump(rejection_counts, "oracle_unreliable")
            continue
        label = str(row.get("difficulty"))
        score = row.get("models", {}).get("ride", {}).get("structure_score", {}).get("mean")
        if label not in eligible or score is None or float(score) < cfg.min_structure_score:
            _bump(rejection_counts, "trivial_or_unscored")
            continue
        eligible[label].append((float(score), target_id, target, row))

    for values in eligible.values():
        values.sort(key=lambda item: (item[0], item[1]))
    easy_limit = min(len(eligible["easy"]), round(cfg.limit * cfg.easy_anchor_fraction))
    difficult_limit = max(0, cfg.limit - easy_limit)
    difficult = sorted([*eligible["hard"], *eligible["medium"]], key=lambda item: (item[0], item[1]))
    chosen = [*difficult[:difficult_limit], *eligible["easy"][:easy_limit]]
    if len(chosen) < cfg.limit:
        chosen_ids = {item[1] for item in chosen}
        remainder = sorted(
            [item for values in eligible.values() for item in values if item[1] not in chosen_ids],
            key=lambda item: (item[0], item[1]),
        )
        chosen.extend(remainder[: cfg.limit - len(chosen)])

    targets: list[dict[str, Any]] = []
    for score, _, target, row in chosen[: cfg.limit]:
        copied = deepcopy(target)
        metadata = dict(copied.get("metadata", {}))
        oracle = row["oracle"]
        oracle_values = oracle["values"]
        ride_baseline = row["models"]["ride"]
        metadata.update(
            {
                "curriculum_source": source_label,
                "difficulty": row["difficulty"],
                "selection_reason": "easy_anchor" if row["difficulty"] == "easy" else "frozen_ride_underperforming",
                "length_bucket": length_bucket(int(metadata["length"])),
                "oracle_reliability": oracle,
            }
        )
        copied["metadata"] = metadata
        copied["calibration"] = {
            "status": "ok",
            "backend": "rhofold-plus-native-refold",
            "rmsd": oracle_values["rmsd"],
            "tm_score": oracle_values["tm"],
            "gdt": oracle_values["gdt"],
            "reliability": oracle,
        }
        copied["frozen_baseline"] = {
            "status": "ok",
            "backend": "frozen-ride-structural-evaluation",
            "structure_score": score,
            "success_rate": ride_baseline["success_rate"],
            "good_rate": ride_baseline["good_rate"],
            "metrics": ride_baseline["metrics"],
        }
        copied["curriculum_score"] = score
        targets.append(copied)

    return _derived_pool(
        source_pool,
        targets,
        selection="oracle_reliable_frozen_ride_curriculum",
        config=asdict(cfg),
        extra_summary={
            "source": source_label,
            "difficulty_counts": _counts(t["metadata"]["difficulty"] for t in targets),
            "length_counts": _counts(t["metadata"]["length_bucket"] for t in targets),
            "rejected": rejection_counts,
        },
    )


def select_rna3db_curriculum_pool(
    source_pool: Mapping[str, Any],
    config: RNA3DBSelectionConfig | None = None,
) -> dict[str, Any]:
    """Select novel RNA3DB chains by structural quality with a recent-data floor."""

    cfg = config or RNA3DBSelectionConfig()
    eligible: list[tuple[tuple[float, int, int, str], Mapping[str, Any]]] = []
    rejected: dict[str, int] = {}
    for target in source_pool.get("targets", []):
        metadata = target.get("metadata", {})
        provenance = metadata.get("rna3db", {})
        try:
            resolution = float(provenance["resolution"])
            release_date = str(provenance["release_date"])
            terminal_trim = int(provenance.get("terminal_trim_left", 0)) + int(provenance.get("terminal_trim_right", 0))
        except (KeyError, TypeError, ValueError):
            _bump(rejected, "missing_quality_metadata")
            continue
        if resolution > cfg.max_resolution:
            _bump(rejected, "resolution_too_low")
            continue
        if terminal_trim > cfg.max_terminal_trim:
            _bump(rejected, "terminal_trim_too_large")
            continue
        release_ordinal = _date_ordinal(release_date)
        eligible.append(((resolution, terminal_trim, -release_ordinal, _pool_target_id(target)), target))

    eligible.sort(key=lambda item: item[0])
    recent = [item for item in eligible if str(item[1]["metadata"]["rna3db"]["release_date"]) >= cfg.recent_cutoff]
    recent_limit = min(len(recent), round(cfg.limit * cfg.recent_minimum_fraction))
    chosen = recent[:recent_limit]
    chosen_ids = {_pool_target_id(item[1]) for item in chosen}
    chosen.extend(item for item in eligible if _pool_target_id(item[1]) not in chosen_ids)

    targets: list[dict[str, Any]] = []
    for _, target in chosen[: cfg.limit]:
        copied = deepcopy(target)
        metadata = dict(copied.get("metadata", {}))
        provenance = metadata["rna3db"]
        metadata.update(
            {
                "curriculum_source": "rna3db_novel",
                "difficulty": "new",
                "selection_reason": "novel_high_quality_structure",
                "length_bucket": length_bucket(int(metadata["length"])),
                "novelty_screen": {
                    "processed_exact_sequence": "excluded",
                    "processed_near_homology": "excluded",
                    "threshold": source_pool.get("resolved_config", {}).get("homology_threshold"),
                },
                "quality_screen": {
                    "resolution": float(provenance["resolution"]),
                    "release_date": str(provenance["release_date"]),
                    "terminal_trim": int(provenance.get("terminal_trim_left", 0))
                    + int(provenance.get("terminal_trim_right", 0)),
                },
            }
        )
        copied["metadata"] = metadata
        targets.append(copied)

    return _derived_pool(
        source_pool,
        targets,
        selection="novel_high_quality_rna3db",
        config=asdict(cfg),
        extra_summary={
            "source": "rna3db_novel",
            "recent_selected": sum(
                str(target["metadata"]["rna3db"]["release_date"]) >= cfg.recent_cutoff for target in targets
            ),
            "length_counts": _counts(target["metadata"]["length_bucket"] for target in targets),
            "rejected": rejected,
        },
    )


def select_oracle_reliable_pool(
    source_pool: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    config: OracleSelectionConfig | None = None,
) -> dict[str, Any]:
    """Keep targets where RhoFold+ refolds the native sequence reliably."""

    cfg = config or OracleSelectionConfig()
    if not 1 <= cfg.minimum_passed_checks <= 3:
        raise ValueError("minimum_passed_checks must be between 1 and 3")
    evaluated = {str(target.get("target_id", "")): target for target in evaluation.get("targets", [])}
    targets: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for target in source_pool.get("targets", []):
        if cfg.limit is not None and len(targets) >= cfg.limit:
            break
        target_id = _pool_target_id(target)
        row = evaluated.get(target_id)
        if row is None:
            _bump(rejected, "missing_evaluation")
            continue
        metrics = row.get("metrics", {})
        values = {
            "tm": _metric_value(metrics.get("native_reward_c4p_tm_score")),
            "gdt": _metric_value(metrics.get("native_reward_c4p_gdt_ts")),
            "rmsd": _metric_value(metrics.get("native_reward_c4p_rmsd")),
        }
        checks = {
            "tm": values["tm"] is not None and values["tm"] >= cfg.native_tm_threshold,
            "gdt": values["gdt"] is not None and values["gdt"] >= cfg.native_gdt_threshold,
            "rmsd": values["rmsd"] is not None and values["rmsd"] <= cfg.native_rmsd_threshold,
        }
        passed = sum(checks.values())
        if passed < cfg.minimum_passed_checks:
            _bump(rejected, "oracle_unreliable")
            continue
        copied = deepcopy(target)
        metadata = dict(copied.get("metadata", {}))
        reliability = {
            "accepted": True,
            "reliable": passed >= 2,
            "tier": "reliable" if passed >= 2 else "usable",
            "passed": passed,
            "required": cfg.minimum_passed_checks,
            "checks": checks,
            "values": values,
        }
        metadata["oracle_reliability"] = reliability
        copied["metadata"] = metadata
        copied["calibration"] = {
            "status": "ok",
            "backend": "rhofold-plus-native-refold",
            "rmsd": values["rmsd"],
            "tm_score": values["tm"],
            "gdt": values["gdt"],
            "reliability": reliability,
        }
        targets.append(copied)
        source = str(metadata.get("curriculum_source", metadata.get("source_split", "unknown")))
        _bump(source_counts, source)

    if len(targets) < cfg.minimum_targets:
        raise ValueError(f"oracle-reliable curriculum has {len(targets)} targets, below required minimum {cfg.minimum_targets}")
    return _derived_pool(
        source_pool,
        targets,
        selection="rhofold_plus_native_oracle_reliable",
        config=asdict(cfg),
        extra_summary={
            "source_counts": dict(sorted(source_counts.items())),
            "oracle_tier_counts": _counts(target["metadata"]["oracle_reliability"]["tier"] for target in targets),
            "length_counts": _counts(length_bucket(int(target["metadata"]["length"])) for target in targets),
            "rejected": dict(sorted(rejected.items())),
        },
    )


def merge_curriculum_pools(
    pools: Sequence[Mapping[str, Any]],
    *,
    minimum_targets: int = 500,
    limit: int | None = None,
) -> dict[str, Any]:
    """Merge derived pools with exact-sequence and structure-hash deduplication."""

    targets: list[dict[str, Any]] = []
    sequences: set[str] = set()
    structure_hashes: set[str] = set()
    source_counts: dict[str, int] = {}
    rejected: dict[str, int] = {}
    source_targets = [list(pool.get("targets", [])) for pool in pools]
    max_source_size = max((len(values) for values in source_targets), default=0)
    for index in range(max_source_size):
        for values in source_targets:
            if index >= len(values):
                continue
            if limit is not None and len(targets) >= limit:
                break
            target = values[index]
            metadata = target.get("metadata", {})
            sequence = str(metadata.get("sequence", ""))
            structure_hash = str(metadata.get("structure_hash", ""))
            if sequence in sequences:
                _bump(rejected, "exact_sequence_duplicate")
                continue
            if structure_hash and structure_hash in structure_hashes:
                _bump(rejected, "structure_hash_duplicate")
                continue
            copied = deepcopy(target)
            targets.append(copied)
            sequences.add(sequence)
            if structure_hash:
                structure_hashes.add(structure_hash)
            source = str(metadata.get("curriculum_source", metadata.get("source_split", "unknown")))
            _bump(source_counts, source)
        if limit is not None and len(targets) >= limit:
            break
    if len(targets) < minimum_targets:
        raise ValueError(f"merged curriculum has {len(targets)} targets, below required minimum {minimum_targets}")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _latest_created_at(pools),
        "resolved_config": {
            "minimum_targets": minimum_targets,
            "limit": limit,
            "source_pool_count": len(pools),
        },
        "summary": {
            "selected": len(targets),
            "source_counts": dict(sorted(source_counts.items())),
            "difficulty_counts": _counts(t.get("metadata", {}).get("difficulty", "unknown") for t in targets),
            "length_counts": _counts(length_bucket(int(t["metadata"]["length"])) for t in targets),
            "rejected": dict(sorted(rejected.items())),
        },
        "targets": targets,
    }


def _metric_value(metric: Any) -> float | None:
    if not isinstance(metric, Mapping) or metric.get("status") != "ok" or metric.get("value") is None:
        return None
    try:
        return float(metric["value"])
    except (TypeError, ValueError):
        return None


def _fractional_quotas(limit: int, fractions: Mapping[str, float]) -> dict[str, int]:
    raw = {key: max(0.0, float(value)) * limit for key, value in fractions.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = max(0, limit - sum(quotas.values()))
    for key in sorted(raw, key=lambda item: (raw[item] - quotas[item], item), reverse=True)[:remaining]:
        quotas[key] += 1
    return quotas


def _with_curriculum_metadata(target: Mapping[str, Any], screen: Mapping[str, Any]) -> dict[str, Any]:
    copied = deepcopy(target)
    metadata = dict(copied.get("metadata", {}))
    metadata["curriculum_source"] = screen["source"]
    metadata["curriculum_screen"] = dict(screen)
    metadata["curriculum_class"] = "ride_pretrain_underperforming"
    metadata["selection_reason"] = "low_frozen_ride_sequence_recovery"
    copied["metadata"] = metadata
    copied["frozen_baseline"] = {
        "status": "ok",
        "backend": "frozen-ride-sequence-screen",
        "sequence_recovery": screen["sequence_recovery_mean"],
        "internal_diversity": screen["internal_diversity"],
        "n_samples": screen["n_samples"],
    }
    return copied


def _derived_pool(
    source_pool: Mapping[str, Any],
    targets: list[dict[str, Any]],
    *,
    selection: str,
    config: Mapping[str, Any],
    extra_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": source_pool.get("created_at_utc", "unknown"),
        "resolved_config": {"selection": selection, **dict(config)},
        "summary": {
            "source_targets": len(source_pool.get("targets", [])),
            "selected": len(targets),
            **dict(extra_summary),
        },
        "targets": targets,
    }


def _pool_target_id(target: Mapping[str, Any]) -> str:
    return str(target.get("metadata", {}).get("target_id", target.get("target_id", "")))


def _counts(values: Sequence[Any] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        _bump(counts, str(value))
    return dict(sorted(counts.items()))


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _date_ordinal(value: str) -> int:
    try:
        return date.fromisoformat(value[:10]).toordinal()
    except ValueError:
        return 0


def _latest_created_at(pools: Sequence[Mapping[str, Any]]) -> str:
    values = [str(pool.get("created_at_utc", "")) for pool in pools if pool.get("created_at_utc")]
    return max(values) if values else "unknown"


__all__ = [
    "RecoveryScreenConfig",
    "OracleSelectionConfig",
    "RNA3DBSelectionConfig",
    "StructuralSelectionConfig",
    "load_json",
    "merge_curriculum_pools",
    "select_low_recovery_pool",
    "select_oracle_reliable_pool",
    "select_rna3db_curriculum_pool",
    "select_structural_curriculum_pool",
]
