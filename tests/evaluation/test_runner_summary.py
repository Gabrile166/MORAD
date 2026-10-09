from __future__ import annotations

import pytest

from src.evaluation.runner import summarize_results


def test_summarize_results_counts_ok_skipped_and_error_statuses() -> None:
    summary = summarize_results(
        [_target("short-target", length_bucket="short", rna_family="RF00001")],
        [
            _candidate("short-target:1", "short-target", sequence_recovery=_ok(1.0)),
            _candidate("short-target:2", "short-target", sequence_recovery=_skipped("missing")),
            _candidate("short-target:3", "short-target", sequence_recovery=_error("bad length")),
        ],
    )

    recovery = summary["candidate_metrics"]["sequence_recovery"]
    assert recovery["count"] == 3
    assert recovery["ok"] == 1
    assert recovery["skipped"] == 1
    assert recovery["error"] == 1
    assert recovery["mean"] == pytest.approx(1.0)
    assert recovery["reason_counts"] == {"bad length": 1, "missing": 1}


def test_summarize_results_groups_candidate_metrics_by_length_bucket() -> None:
    summary = summarize_results(
        [
            _target("short-target", length_bucket="short", rna_family="RF00001"),
            _target("long-target", length_bucket="long", rna_family="RF00002"),
        ],
        [
            _candidate("short-target:1", "short-target", sequence_recovery=_ok(1.0)),
            _candidate("long-target:1", "long-target", sequence_recovery=_ok(0.25)),
        ],
    )

    assert summary["by_length_bucket"]["short"]["sequence_recovery"]["mean"] == pytest.approx(1.0)
    assert summary["by_length_bucket"]["long"]["sequence_recovery"]["mean"] == pytest.approx(0.25)


def test_summarize_results_averages_rfam_success_across_families() -> None:
    summary = summarize_results(
        [
            _target("family-a", length_bucket="short", rna_family="RF00001"),
            _target("family-b", length_bucket="medium", rna_family="RF00002"),
        ],
        [
            _candidate("family-a:1", "family-a", rfam_family_success=_ok(1.0)),
            _candidate("family-a:2", "family-a", rfam_family_success=_ok(0.0)),
            _candidate("family-b:1", "family-b", rfam_family_success=_ok(1.0)),
            _candidate("family-b:2", "family-b", rfam_family_success=_skipped("database_missing")),
        ],
    )

    rfam = summary["rfam_success_average_across_families"]
    assert rfam["status"] == "ok"
    assert rfam["family_means"] == {"RF00001": pytest.approx(0.5), "RF00002": pytest.approx(1.0)}
    assert rfam["value"] == pytest.approx(0.75)


def _target(target_id: str, *, length_bucket: str, rna_family: str) -> dict:
    return {
        "target_id": target_id,
        "length_bucket": length_bucket,
        "rna_type": "ncRNA",
        "rna_family": rna_family,
        "metrics": {
            "internal_diversity": _ok(0.2),
            "native_rhofold_c1prime_tm_score": _skipped("tertiary disabled"),
        },
    }


def _candidate(
    candidate_id: str,
    target_id: str,
    *,
    sequence_recovery: dict | None = None,
    secondary_structure_f1: dict | None = None,
    rfam_family_success: dict | None = None,
    rhofold_c1prime_tm_score: dict | None = None,
    drfold_c1prime_tm_score: dict | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "target_id": target_id,
        "metrics": {
            "sequence_recovery": sequence_recovery or _ok(0.5),
            "secondary_structure_f1": secondary_structure_f1 or _ok(0.4),
            "rfam_family_success": rfam_family_success or _ok(1.0),
            "rhofold_c1prime_tm_score": rhofold_c1prime_tm_score or _skipped("tertiary disabled"),
            "drfold_c1prime_tm_score": drfold_c1prime_tm_score or _skipped("DRFold disabled"),
        },
    }


def _ok(value: float) -> dict:
    return {"status": "ok", "value": value, "reason": None}


def _skipped(reason: str) -> dict:
    return {"status": "skipped", "value": None, "reason": reason}


def _error(reason: str) -> dict:
    return {"status": "error", "value": None, "reason": reason}
