from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from src.evaluation.difficulty import (
    DifficultyConfig,
    build_medium_hard_pool,
    classify_difficulty,
    write_difficulty_outputs,
    write_medium_hard_pool,
)


def test_target_alignment_error_reports_both_sides() -> None:
    with pytest.raises(ValueError, match="target_id mismatch"):
        classify_difficulty(_evaluation(["a"]), _evaluation(["b"]))


def test_missing_candidate_metrics_do_not_count_as_success() -> None:
    report = classify_difficulty(
        _evaluation(["t"], candidate_specs={"t": [{}]}),
        _evaluation(["t"], candidate_specs={"t": [_candidate_metrics(tm=0.7, gdt=0.7, rmsd=1.0, good=1.0)]}),
    )

    ride = report["targets"][0]["models"]["ride"]
    assert ride["success_rate"] == pytest.approx(0.0)
    assert ride["structure_score"]["mean"] is None
    assert report["targets"][0]["reason"] == "ride_trivial_or_failed"


def test_oracle_gate_requires_two_of_three_native_checks() -> None:
    unreliable = _evaluation(["t"], native={"t": {"tm": 0.44, "gdt": 0.49, "rmsd": 1.0}})

    report = classify_difficulty(unreliable, _evaluation(["t"]))

    row = report["targets"][0]
    assert row["difficulty"] == "oracle_unreliable"
    assert row["oracle"]["passed"] == 1
    assert row["reason"] == "oracle_unreliable"


def test_quantile_layering_assigns_easy_medium_and_hard() -> None:
    report = classify_difficulty(
        _evaluation(
            ["hard", "medium", "easy"],
            candidate_specs={
                "hard": [_candidate_metrics(tm=0.10, gdt=0.10, rmsd=8.0, good=0.0)],
                "medium": [_candidate_metrics(tm=0.50, gdt=0.50, rmsd=5.0, good=0.0)],
                "easy": [_candidate_metrics(tm=0.90, gdt=0.90, rmsd=1.0, good=1.0)],
            },
        ),
        _evaluation(["hard", "medium", "easy"]),
    )

    labels = {row["target_id"]: row["difficulty"] for row in report["targets"]}
    assert labels == {"hard": "hard", "medium": "medium", "easy": "easy"}


def test_medium_hard_selection_requires_reliable_nontrivial_partial_success_and_positive_gap() -> None:
    ride = _evaluation(
        ["hard", "both_failed", "easy", "negative_gap"],
        candidate_specs={
            "hard": [_candidate_metrics(tm=0.30, gdt=0.30, rmsd=7.0, good=0.0)],
            "both_failed": [{}],
            "easy": [_candidate_metrics(tm=0.90, gdt=0.90, rmsd=1.0, good=1.0)],
            "negative_gap": [_candidate_metrics(tm=0.45, gdt=0.45, rmsd=5.0, good=1.0)],
        },
    )
    ribo = _evaluation(
        ["hard", "both_failed", "easy", "negative_gap"],
        candidate_specs={
            "hard": [_candidate_metrics(tm=0.60, gdt=0.60, rmsd=3.0, good=1.0)],
            "both_failed": [{}],
            "easy": [_candidate_metrics(tm=0.95, gdt=0.95, rmsd=0.5, good=1.0)],
            "negative_gap": [_candidate_metrics(tm=0.20, gdt=0.20, rmsd=8.0, good=1.0)],
        },
    )

    report = classify_difficulty(ride, ribo, DifficultyConfig(min_ride_score=0.01))
    rows = {row["target_id"]: row for row in report["targets"]}

    assert report["medium_hard_target_ids"] == ["hard"]
    assert rows["hard"]["selected"] is True
    assert rows["both_failed"]["reason"] == "ride_trivial_or_failed"
    assert rows["easy"]["reason"] == "difficulty_easy"
    assert rows["negative_gap"]["reason"] == "non_positive_headroom"


def test_medium_hard_selection_requires_at_least_one_good_ribodiffusion_sample() -> None:
    ride = _evaluation(
        ["t"],
        candidate_specs={"t": [_candidate_metrics(tm=0.20, gdt=0.20, rmsd=8.0, good=0.0)]},
    )
    ribo = _evaluation(
        ["t"],
        candidate_specs={"t": [_candidate_metrics(tm=0.60, gdt=0.60, rmsd=3.0, good=0.0)]},
    )

    report = classify_difficulty(ride, ribo, DifficultyConfig(min_ride_score=0.01))

    assert report["medium_hard_target_ids"] == []
    assert report["targets"][0]["reason"] == "ribodiffusion_good_rate_below_threshold"


def test_outputs_write_json_csv_and_markdown(tmp_path: Path) -> None:
    report = classify_difficulty(_evaluation(["t"]), _evaluation(["t"]))

    paths = write_difficulty_outputs(report, tmp_path)

    assert json.loads(Path(paths["json"]).read_text(encoding="utf-8"))["schema_version"] == "ride_rl_difficulty.v1"
    assert "target_id,difficulty,selected,reason" in Path(paths["csv"]).read_text(encoding="utf-8")
    assert "# RIDE RL Difficulty Calibration" in Path(paths["md"]).read_text(encoding="utf-8")


def test_medium_hard_pool_preserves_target_and_adds_metadata(tmp_path: Path) -> None:
    ride = _evaluation(
        ["t"],
        candidate_specs={"t": [_candidate_metrics(tm=0.30, gdt=0.30, rmsd=7.0, good=0.0)]},
    )
    ribo = _evaluation(
        ["t"],
        candidate_specs={"t": [_candidate_metrics(tm=0.70, gdt=0.70, rmsd=2.0, good=1.0)]},
    )
    report = classify_difficulty(ride, ribo, DifficultyConfig(min_ride_score=0.01))
    source_pool = {
        "schema_version": "ride_rl_target_pool.v1",
        "summary": {"selected": 2},
        "targets": [
            {"metadata": {"target_id": "t", "dataset_index": 1}, "raw_record": {"sequence": "AUGC"}},
            {"metadata": {"target_id": "other", "dataset_index": 2}},
        ],
    }

    pool = build_medium_hard_pool(
        source_pool,
        report,
        ride_evaluation_path="ride/evaluation.json",
        ribodiffusion_evaluation_path="ribo/evaluation.json",
    )

    assert pool["schema_version"] == "ride_rl_target_pool.v1"
    assert len(pool["targets"]) == 1
    target = pool["targets"][0]
    assert target["raw_record"] == {"sequence": "AUGC"}
    assert target["metadata"]["difficulty"] == "hard"
    assert target["metadata"]["difficulty_calibration"]["learnability"]["score_gap"] > 0
    assert target["metadata"]["frozen_baseline"]["ride"]["structure_score"]["mean"] is not None
    assert target["frozen_baseline"] == target["metadata"]["frozen_baseline"]
    assert target["calibration"] == target["metadata"]["difficulty_calibration"]


def test_medium_hard_pool_writer_uses_versioned_paths(tmp_path: Path) -> None:
    saved = {}
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = types.SimpleNamespace(save=lambda payload, path: saved.update({"payload": payload, "path": path}))
    try:
        paths = write_medium_hard_pool({"summary": {"selected": 1}, "targets": []}, tmp_path)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch

    assert Path(paths["pool"]) == tmp_path / "ride_rl_target_pool.v1.pt"
    assert Path(paths["summary"]) == tmp_path / "ride_rl_target_pool.v1.summary.json"
    assert saved["path"] == tmp_path / "ride_rl_target_pool.v1.pt"
    assert json.loads((tmp_path / "ride_rl_target_pool.v1.summary.json").read_text(encoding="utf-8")) == {"selected": 1}


def _evaluation(
    target_ids: list[str],
    *,
    native: dict[str, dict[str, float]] | None = None,
    candidate_specs: dict[str, list[dict]] | None = None,
) -> dict:
    native = native or {}
    candidate_specs = candidate_specs or {}
    targets = []
    candidates = []
    for target_id in target_ids:
        native_values = native.get(target_id, {"tm": 0.7, "gdt": 0.7, "rmsd": 1.0})
        targets.append(
            {
                "target_id": target_id,
                "metrics": {
                    "native_reward_c4p_tm_score": _ok(native_values["tm"]),
                    "native_reward_c4p_gdt_ts": _ok(native_values["gdt"]),
                    "native_reward_c4p_rmsd": _ok(native_values["rmsd"]),
                },
            }
        )
        specs = candidate_specs.get(target_id, [_candidate_metrics(tm=0.5, gdt=0.5, rmsd=5.0, good=0.0)])
        for index, metrics in enumerate(specs):
            candidates.append({"target_id": target_id, "candidate_id": f"{target_id}:{index}", "metrics": metrics})
    return {"targets": targets, "candidates": candidates}


def _candidate_metrics(*, tm: float, gdt: float, rmsd: float, good: float, raw: float = 1.0) -> dict:
    return {
        "reward_c4p_tm_score": _ok(tm),
        "reward_c4p_gdt_ts": _ok(gdt),
        "reward_c4p_rmsd": _ok(rmsd),
        "reward_raw_score": _ok(raw),
        "reward_good": _ok(good),
    }


def _ok(value: float) -> dict:
    return {"status": "ok", "value": value, "reason": None}
