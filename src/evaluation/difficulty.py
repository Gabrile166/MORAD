"""Reward-aligned difficulty calibration for RIDE RL target selection."""

from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


REWARD_METRICS = (
    "reward_c4p_tm_score",
    "reward_c4p_gdt_ts",
    "reward_c4p_rmsd",
    "reward_raw_score",
    "reward_good",
)
NATIVE_REWARD_METRICS = tuple(f"native_{name}" for name in REWARD_METRICS)


@dataclass(frozen=True)
class DifficultyConfig:
    native_tm_threshold: float = 0.45
    native_gdt_threshold: float = 0.50
    native_rmsd_threshold: float = 2.0
    rmsd_cap: float = 10.0
    hard_quantile: float = 1.0 / 3.0
    easy_quantile: float = 2.0 / 3.0
    min_ride_score: float = 0.05
    min_ride_success_rate: float = 0.0
    min_ribo_success_rate: float = 0.0
    min_ribo_good_rate: float = 0.125
    min_score_gap: float = 0.0
    min_good_rate_gap: float = 0.0


def load_evaluation_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def classify_difficulty(
    ride_evaluation: Mapping[str, Any],
    ribodiffusion_evaluation: Mapping[str, Any],
    config: DifficultyConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DifficultyConfig()
    ride_targets = _index_evaluation(ride_evaluation, "ride")
    ribo_targets = _index_evaluation(ribodiffusion_evaluation, "ribodiffusion")
    _validate_aligned_targets(ride_targets, ribo_targets)

    rows: list[dict[str, Any]] = []
    ride_scores_for_quantiles: list[float] = []
    for target_id in sorted(ride_targets):
        ride_record = ride_targets[target_id]
        ribo_record = ribo_targets[target_id]
        oracle = _oracle_reliability(ride_record["target"], cfg)
        ride_stats = _aggregate_target(ride_record["candidates"], cfg)
        ribo_stats = _aggregate_target(ribo_record["candidates"], cfg)
        if oracle["reliable"] and ride_stats["structure_score"]["count"] > 0:
            ride_scores_for_quantiles.append(float(ride_stats["structure_score"]["mean"]))
        rows.append(
            {
                "target_id": target_id,
                "oracle": oracle,
                "models": {
                    "ride": ride_stats,
                    "ribodiffusion": ribo_stats,
                },
            }
        )

    thresholds = _difficulty_thresholds(ride_scores_for_quantiles, cfg)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        ride_stats = row["models"]["ride"]
        ribo_stats = row["models"]["ribodiffusion"]
        label = _difficulty_label(row["oracle"], ride_stats, thresholds)
        learnability = _learnability(ride_stats, ribo_stats)
        decision = _medium_hard_decision(label, row["oracle"], ride_stats, ribo_stats, learnability, cfg)
        enriched = {
            **row,
            "difficulty": label,
            "learnability": learnability,
            "selected": decision["selected"],
            "reason": decision["reason"],
        }
        if decision["selected"]:
            selected.append(enriched)
        else:
            rejected.append(enriched)

    return {
        "schema_version": "ride_rl_difficulty.v1",
        "config": asdict(cfg),
        "difficulty_thresholds": thresholds,
        "summary": {
            "targets": len(rows),
            "selected": len(selected),
            "rejected": len(rejected),
            "difficulty_counts": _counts(row["difficulty"] for row in [*selected, *rejected]),
            "rejection_counts": _counts(row["reason"] for row in rejected),
        },
        "targets": [*selected, *rejected],
        "medium_hard_target_ids": [row["target_id"] for row in selected],
    }


def write_difficulty_outputs(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "difficulty.json"
    csv_path = root / "difficulty.csv"
    md_path = root / "difficulty.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(report, csv_path)
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def build_medium_hard_pool(
    source_pool: Mapping[str, Any],
    difficulty_report: Mapping[str, Any],
    *,
    ride_evaluation_path: str | Path,
    ribodiffusion_evaluation_path: str | Path,
) -> dict[str, Any]:
    selected_by_id = {row["target_id"]: row for row in difficulty_report.get("targets", []) if row.get("selected")}
    targets: list[dict[str, Any]] = []
    for target in source_pool.get("targets", []):
        target_id = _target_id_from_pool_target(target)
        if target_id not in selected_by_id:
            continue
        copied = deepcopy(target)
        metadata = dict(copied.get("metadata", {}))
        selected = selected_by_id[target_id]
        metadata["difficulty"] = selected["difficulty"]
        metadata["difficulty_reason"] = selected["reason"]
        metadata["difficulty_calibration"] = {
            "oracle": selected["oracle"],
            "learnability": selected["learnability"],
            "thresholds": difficulty_report.get("difficulty_thresholds", {}),
        }
        frozen_baseline = {
            "ride": selected["models"]["ride"],
            "ribodiffusion": selected["models"]["ribodiffusion"],
        }
        calibration = {
            "oracle": selected["oracle"],
            "learnability": selected["learnability"],
            "thresholds": difficulty_report.get("difficulty_thresholds", {}),
        }
        metadata["frozen_baseline"] = frozen_baseline
        metadata["difficulty_provenance"] = {
            "source_pool_schema_version": source_pool.get("schema_version"),
            "ride_evaluation": str(ride_evaluation_path),
            "ribodiffusion_evaluation": str(ribodiffusion_evaluation_path),
        }
        copied["frozen_baseline"] = frozen_baseline
        copied["calibration"] = calibration
        copied["metadata"] = metadata
        targets.append(copied)

    summary = dict(source_pool.get("summary", {}))
    summary.update(
        {
            "selected": len(targets),
            "source_targets": len(source_pool.get("targets", [])),
            "selection": "medium_hard",
            "missing_selected_target_ids": sorted(set(selected_by_id) - {_target_id_from_pool_target(t) for t in targets}),
        }
    )
    return {
        **{key: deepcopy(value) for key, value in source_pool.items() if key not in {"targets", "summary"}},
        "schema_version": "ride_rl_target_pool.v1",
        "summary": summary,
        "targets": targets,
    }


def write_medium_hard_pool(pool: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    import torch

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pool_path = root / "ride_rl_target_pool.v1.pt"
    summary_path = root / "ride_rl_target_pool.v1.summary.json"
    torch.save(dict(pool), pool_path)
    summary_path.write_text(json.dumps(pool.get("summary", {}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"pool": str(pool_path), "summary": str(summary_path)}


def _index_evaluation(evaluation: Mapping[str, Any], model_name: str) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for target in evaluation.get("targets", []):
        target_id = target.get("target_id")
        if target_id is None:
            raise ValueError(f"{model_name} evaluation target is missing target_id")
        targets[str(target_id)] = {"target": target, "candidates": []}
    for candidate in evaluation.get("candidates", []):
        target_id = candidate.get("target_id")
        if target_id is None:
            raise ValueError(f"{model_name} evaluation candidate is missing target_id")
        target_id = str(target_id)
        targets.setdefault(target_id, {"target": {"target_id": target_id, "metrics": {}}, "candidates": []})
        targets[target_id]["candidates"].append(candidate)
    return targets


def _validate_aligned_targets(ride: Mapping[str, Any], ribo: Mapping[str, Any]) -> None:
    ride_ids = set(ride)
    ribo_ids = set(ribo)
    if ride_ids != ribo_ids:
        raise ValueError(
            "evaluation target_id mismatch: "
            f"ride_only={sorted(ride_ids - ribo_ids)}, ribodiffusion_only={sorted(ribo_ids - ride_ids)}"
        )


def _oracle_reliability(target: Mapping[str, Any], config: DifficultyConfig) -> dict[str, Any]:
    metrics = target.get("metrics", {})
    checks = {
        "tm": _metric_value(metrics.get("native_reward_c4p_tm_score")) >= config.native_tm_threshold,
        "gdt": _metric_value(metrics.get("native_reward_c4p_gdt_ts")) >= config.native_gdt_threshold,
        "rmsd": _metric_value(metrics.get("native_reward_c4p_rmsd")) <= config.native_rmsd_threshold,
    }
    values = {
        "tm": _metric_value(metrics.get("native_reward_c4p_tm_score")),
        "gdt": _metric_value(metrics.get("native_reward_c4p_gdt_ts")),
        "rmsd": _metric_value(metrics.get("native_reward_c4p_rmsd")),
    }
    passed = sum(1 for value in checks.values() if value)
    return {
        "reliable": passed >= 2,
        "passed": passed,
        "required": 2,
        "checks": checks,
        "values": values,
    }


def _aggregate_target(candidates: Sequence[Mapping[str, Any]], config: DifficultyConfig) -> dict[str, Any]:
    metric_stats = {
        name: _aggregate_metric([candidate.get("metrics", {}).get(name) for candidate in candidates], best_min=name.endswith("_rmsd"))
        for name in REWARD_METRICS
    }
    scores = [
        score
        for candidate in candidates
        for score in [_candidate_structure_score(candidate, config)]
        if score is not None
    ]
    score_stats = _stats(scores)
    candidate_count = len(candidates)
    success_count = len(scores)
    return {
        "candidate_count": candidate_count,
        "success_count": success_count,
        "success_rate": success_count / candidate_count if candidate_count else 0.0,
        "good_rate": metric_stats["reward_good"]["mean"],
        "metrics": metric_stats,
        "structure_score": score_stats,
    }


def _aggregate_metric(metrics: Sequence[Any], *, best_min: bool = False) -> dict[str, Any]:
    values = [_metric_value(metric) for metric in metrics]
    ok_values = [value for value in values if math.isfinite(value)]
    stats = _stats(ok_values, best_min=best_min)
    stats.update(
        {
            "count": len(metrics),
            "ok": len(ok_values),
            "success_rate": len(ok_values) / len(metrics) if metrics else 0.0,
            "missing": len(metrics) - len(ok_values),
        }
    )
    return stats


def _candidate_structure_score(candidate: Mapping[str, Any], config: DifficultyConfig) -> float | None:
    metrics = candidate.get("metrics", {})
    tm = _metric_value(metrics.get("reward_c4p_tm_score"))
    gdt = _metric_value(metrics.get("reward_c4p_gdt_ts"))
    rmsd = _metric_value(metrics.get("reward_c4p_rmsd"))
    good = _metric_value(metrics.get("reward_good"))
    if not all(math.isfinite(value) for value in (tm, gdt, rmsd, good)):
        return None
    rmsd_score = max(0.0, min(1.0, 1.0 - (rmsd / config.rmsd_cap)))
    parts = [max(0.0, min(1.0, tm)), max(0.0, min(1.0, gdt)), rmsd_score, max(0.0, min(1.0, good))]
    return mean(parts)


def _stats(values: Sequence[float], *, best_min: bool = False) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "best": None, "std": None}
    avg = mean(values)
    variance = mean([(value - avg) ** 2 for value in values])
    return {
        "count": len(values),
        "mean": avg,
        "median": median(values),
        "best": min(values) if best_min else max(values),
        "std": math.sqrt(variance),
    }


def _difficulty_thresholds(scores: Sequence[float], config: DifficultyConfig) -> dict[str, Any]:
    if not scores:
        return {"hard_max": None, "easy_min": None}
    return {
        "hard_max": _quantile(scores, config.hard_quantile),
        "easy_min": _quantile(scores, config.easy_quantile),
    }


def _difficulty_label(oracle: Mapping[str, Any], ride_stats: Mapping[str, Any], thresholds: Mapping[str, Any]) -> str:
    if not oracle.get("reliable"):
        return "oracle_unreliable"
    score = ride_stats.get("structure_score", {}).get("mean")
    if score is None:
        return "hard"
    hard_max = thresholds.get("hard_max")
    easy_min = thresholds.get("easy_min")
    if hard_max is not None and score <= hard_max:
        return "hard"
    if easy_min is not None and score >= easy_min:
        return "easy"
    return "medium"


def _learnability(ride_stats: Mapping[str, Any], ribo_stats: Mapping[str, Any]) -> dict[str, Any]:
    ride_score = ride_stats.get("structure_score", {}).get("mean")
    ribo_score = ribo_stats.get("structure_score", {}).get("mean")
    ride_good = float(ride_stats.get("good_rate") or 0.0)
    ribo_good = float(ribo_stats.get("good_rate") or 0.0)
    return {
        "score_gap": None if ride_score is None or ribo_score is None else float(ribo_score) - float(ride_score),
        "good_rate_gap": ribo_good - ride_good,
    }


def _medium_hard_decision(
    label: str,
    oracle: Mapping[str, Any],
    ride_stats: Mapping[str, Any],
    ribo_stats: Mapping[str, Any],
    learnability: Mapping[str, Any],
    config: DifficultyConfig,
) -> dict[str, Any]:
    if not oracle.get("reliable"):
        return {"selected": False, "reason": "oracle_unreliable"}
    if label not in {"medium", "hard"}:
        return {"selected": False, "reason": f"difficulty_{label}"}
    ride_score = ride_stats.get("structure_score", {}).get("mean")
    if ride_score is None or float(ride_score) <= config.min_ride_score:
        return {"selected": False, "reason": "ride_trivial_or_failed"}
    if float(ride_stats.get("success_rate") or 0.0) <= config.min_ride_success_rate:
        return {"selected": False, "reason": "ride_success_rate_below_threshold"}
    if float(ribo_stats.get("success_rate") or 0.0) <= config.min_ribo_success_rate:
        return {"selected": False, "reason": "ribodiffusion_failed"}
    if float(ribo_stats.get("good_rate") or 0.0) < config.min_ribo_good_rate:
        return {"selected": False, "reason": "ribodiffusion_good_rate_below_threshold"}
    score_gap = learnability.get("score_gap")
    good_gap = float(learnability.get("good_rate_gap") or 0.0)
    if score_gap is None:
        return {"selected": False, "reason": "missing_headroom_metrics"}
    if float(score_gap) <= config.min_score_gap and good_gap <= config.min_good_rate_gap:
        return {"selected": False, "reason": "non_positive_headroom"}
    return {"selected": True, "reason": "medium_hard_headroom_positive"}


def _metric_value(metric: Any) -> float:
    if not isinstance(metric, Mapping) or metric.get("status") != "ok" or metric.get("value") is None:
        return math.nan
    try:
        return float(metric["value"])
    except (TypeError, ValueError):
        return math.nan


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _target_id_from_pool_target(target: Mapping[str, Any]) -> str:
    metadata = target.get("metadata", {})
    return str(metadata.get("target_id", target.get("target_id", "")))


def _write_csv(report: Mapping[str, Any], path: Path) -> None:
    fields = [
        "target_id",
        "difficulty",
        "selected",
        "reason",
        "oracle_reliable",
        "ride_score_mean",
        "ribodiffusion_score_mean",
        "score_gap",
        "ride_good_rate",
        "ribodiffusion_good_rate",
        "good_rate_gap",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report.get("targets", []):
            ride = row["models"]["ride"]
            ribo = row["models"]["ribodiffusion"]
            writer.writerow(
                {
                    "target_id": row["target_id"],
                    "difficulty": row["difficulty"],
                    "selected": row["selected"],
                    "reason": row["reason"],
                    "oracle_reliable": row["oracle"]["reliable"],
                    "ride_score_mean": ride["structure_score"]["mean"],
                    "ribodiffusion_score_mean": ribo["structure_score"]["mean"],
                    "score_gap": row["learnability"]["score_gap"],
                    "ride_good_rate": ride["good_rate"],
                    "ribodiffusion_good_rate": ribo["good_rate"],
                    "good_rate_gap": row["learnability"]["good_rate_gap"],
                }
            )


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# RIDE RL Difficulty Calibration",
        "",
        f"- Targets: {summary.get('targets', 0)}",
        f"- Selected medium/hard: {summary.get('selected', 0)}",
        f"- Difficulty counts: `{summary.get('difficulty_counts', {})}`",
        "",
        "| Target | Difficulty | Selected | Reason | RIDE score | RiboDiffusion score | Score gap |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report.get("targets", []):
        ride_score = row["models"]["ride"]["structure_score"]["mean"]
        ribo_score = row["models"]["ribodiffusion"]["structure_score"]["mean"]
        score_gap = row["learnability"]["score_gap"]
        lines.append(
            f"| `{row['target_id']}` | {row['difficulty']} | {row['selected']} | {row['reason']} | "
            f"{_fmt(ride_score)} | {_fmt(ribo_score)} | {_fmt(score_gap)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"
