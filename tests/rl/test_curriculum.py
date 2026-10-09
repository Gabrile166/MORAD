from __future__ import annotations

import pytest

from src.rl.curriculum import (
    OracleSelectionConfig,
    RecoveryScreenConfig,
    RNA3DBSelectionConfig,
    StructuralSelectionConfig,
    merge_curriculum_pools,
    select_low_recovery_pool,
    select_oracle_reliable_pool,
    select_rna3db_curriculum_pool,
    select_structural_curriculum_pool,
)


def test_low_recovery_screen_selects_worst_with_length_strata() -> None:
    source = _pool([("s1", 40), ("s2", 45), ("m1", 70), ("m2", 80), ("l1", 120), ("l2", 140)])
    evaluation = _evaluation(
        {"s1": 0.1, "s2": 0.9, "m1": 0.2, "m2": 0.8, "l1": 0.3, "l2": 0.7},
        structural=False,
    )

    selected = select_low_recovery_pool(
        source,
        evaluation,
        RecoveryScreenConfig(limit=3, short_fraction=1 / 3, medium_fraction=1 / 3, long_fraction=1 / 3),
    )

    assert {target["metadata"]["target_id"] for target in selected["targets"]} == {"s1", "m1", "l1"}
    assert selected["summary"]["length_counts"] == {"long": 1, "medium": 1, "short": 1}
    assert all(target["frozen_baseline"]["status"] == "ok" for target in selected["targets"])


def test_structural_selection_rejects_unreliable_and_prefers_harder_targets() -> None:
    source = _pool([("hard", 60), ("medium", 60), ("easy", 60), ("bad_oracle", 60)])
    evaluation = _evaluation(
        {"hard": 0.3, "medium": 0.5, "easy": 0.9, "bad_oracle": 0.1},
        structural=True,
        unreliable={"bad_oracle"},
    )

    selected = select_structural_curriculum_pool(
        source,
        evaluation,
        StructuralSelectionConfig(limit=2, min_structure_score=0.01, easy_anchor_fraction=0.0),
    )

    ids = [target["metadata"]["target_id"] for target in selected["targets"]]
    assert ids == ["hard", "medium"]
    assert all(target["metadata"]["difficulty"] in {"hard", "medium"} for target in selected["targets"])
    assert all(target["calibration"]["status"] == "ok" for target in selected["targets"])
    assert selected["targets"][0]["calibration"]["tm_score"] == pytest.approx(0.7)
    assert all(target["frozen_baseline"]["status"] == "ok" for target in selected["targets"])


def test_merge_deduplicates_and_enforces_minimum() -> None:
    left = _pool([("a", 60), ("b", 70)])
    right = _pool([("b2", 70), ("c", 80)])
    right["targets"][0]["metadata"]["sequence"] = left["targets"][1]["metadata"]["sequence"]

    merged = merge_curriculum_pools([left, right], minimum_targets=3)

    assert len(merged["targets"]) == 3
    assert merged["summary"]["rejected"] == {"exact_sequence_duplicate": 1}
    with pytest.raises(ValueError, match="below required minimum"):
        merge_curriculum_pools([left], minimum_targets=3)


def test_merge_interleaves_source_pools() -> None:
    left = _pool([("a", 60), ("b", 61)])
    right = _pool([("c", 62), ("d", 63)])

    merged = merge_curriculum_pools([left, right], minimum_targets=4)

    assert [target["metadata"]["target_id"] for target in merged["targets"]] == ["a", "c", "b", "d"]


def test_rna3db_quality_selection_reserves_recent_and_fills_by_resolution() -> None:
    source = _pool([("old_best", 60), ("old_mid", 60), ("recent_good", 60), ("recent_bad", 60)])
    qualities = {
        "old_best": (1.0, "2020-01-01"),
        "old_mid": (1.5, "2022-01-01"),
        "recent_good": (2.0, "2025-01-01"),
        "recent_bad": (2.5, "2025-02-01"),
    }
    for target in source["targets"]:
        target_id = target["metadata"]["target_id"]
        resolution, release_date = qualities[target_id]
        target["metadata"]["rna3db"] = {
            "resolution": resolution,
            "release_date": release_date,
            "terminal_trim_left": 0,
            "terminal_trim_right": 0,
        }
    source["resolved_config"] = {"homology_threshold": 0.8}

    selected = select_rna3db_curriculum_pool(
        source,
        RNA3DBSelectionConfig(limit=2, recent_cutoff="2024-01-01", recent_minimum_fraction=0.5),
    )

    assert [target["metadata"]["target_id"] for target in selected["targets"]] == ["recent_good", "old_best"]
    assert selected["summary"]["recent_selected"] == 1
    assert all(target["metadata"]["curriculum_source"] == "rna3db_novel" for target in selected["targets"])


def test_oracle_reliable_selection_requires_two_native_checks() -> None:
    source = _pool([("good", 60), ("bad", 60)])
    evaluation = _evaluation({"good": 0.5, "bad": 0.5}, structural=False, unreliable={"bad"})

    selected = select_oracle_reliable_pool(source, evaluation, OracleSelectionConfig(minimum_targets=1))

    assert [target["metadata"]["target_id"] for target in selected["targets"]] == ["good"]
    assert selected["targets"][0]["calibration"]["status"] == "ok"
    assert selected["summary"]["rejected"] == {"oracle_unreliable": 1}


def test_oracle_selection_can_retain_one_check_as_usable_tier() -> None:
    source = _pool([("borderline", 60)])
    evaluation = _evaluation({"borderline": 0.5}, structural=False)
    metrics = evaluation["targets"][0]["metrics"]
    metrics["native_reward_c4p_tm_score"] = _ok(0.5)
    metrics["native_reward_c4p_gdt_ts"] = _ok(0.1)
    metrics["native_reward_c4p_rmsd"] = _ok(9.0)

    selected = select_oracle_reliable_pool(
        source,
        evaluation,
        OracleSelectionConfig(minimum_targets=1, minimum_passed_checks=1),
    )

    assert selected["targets"][0]["metadata"]["oracle_reliability"]["tier"] == "usable"
    assert selected["summary"]["oracle_tier_counts"] == {"usable": 1}


def test_oracle_selection_rejects_invalid_passed_check_count() -> None:
    source = _pool([("target", 60)])
    evaluation = _evaluation({"target": 0.5}, structural=False)

    with pytest.raises(ValueError, match="between 1 and 3"):
        select_oracle_reliable_pool(
            source,
            evaluation,
            OracleSelectionConfig(minimum_targets=1, minimum_passed_checks=0),
        )


def _pool(specs: list[tuple[str, int]]) -> dict:
    targets = []
    for index, (target_id, length) in enumerate(specs):
        targets.append(
            {
                "metadata": {
                    "target_id": target_id,
                    "sequence": ("AUGC" * ((length + 3) // 4))[:length] + ("A" * index),
                    "length": length,
                    "structure_hash": f"hash-{target_id}",
                    "source_split": "train",
                }
            }
        )
    return {
        "schema_version": "ride_rl_target_pool.v1",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "summary": {},
        "targets": targets,
    }


def _evaluation(
    scores: dict[str, float],
    *,
    structural: bool,
    unreliable: set[str] | None = None,
) -> dict:
    unreliable = unreliable or set()
    targets = []
    candidates = []
    for target_id, score in scores.items():
        native = {"tm": 0.7, "gdt": 0.7, "rmsd": 1.0}
        if target_id in unreliable:
            native = {"tm": 0.1, "gdt": 0.1, "rmsd": 9.0}
        targets.append(
            {
                "target_id": target_id,
                "metrics": {
                    "internal_diversity": _ok(0.2),
                    "native_reward_c4p_tm_score": _ok(native["tm"]),
                    "native_reward_c4p_gdt_ts": _ok(native["gdt"]),
                    "native_reward_c4p_rmsd": _ok(native["rmsd"]),
                },
            }
        )
        metrics = {"sequence_recovery": _ok(score)}
        if structural:
            metrics.update(
                {
                    "reward_c4p_tm_score": _ok(score),
                    "reward_c4p_gdt_ts": _ok(score),
                    "reward_c4p_rmsd": _ok((1.0 - score) * 10.0),
                    "reward_raw_score": _ok(score),
                    "reward_good": _ok(1.0 if score >= 0.8 else 0.0),
                }
            )
        candidates.append({"target_id": target_id, "candidate_id": f"{target_id}:0", "metrics": metrics})
    return {"targets": targets, "candidates": candidates}


def _ok(value: float) -> dict:
    return {"status": "ok", "value": value, "reason": None}
