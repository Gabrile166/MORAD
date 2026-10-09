from __future__ import annotations

import csv
import json

import pytest

from src.evaluation.reporting import write_evaluation_artifacts


def test_write_evaluation_artifacts_creates_all_report_files(tmp_path) -> None:
    result = _evaluation_result()

    write_evaluation_artifacts(result, tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "evaluation.json",
        "summary.json",
        "manifest.json",
        "targets.jsonl",
        "candidates.jsonl",
        "metrics.jsonl",
        "summary.csv",
        "REPORT.md",
        "SKIPPED_METRICS.md",
    }
    assert json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))["summary"]["candidate_count"] == 2
    assert len((tmp_path / "targets.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len((tmp_path / "candidates.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_write_evaluation_artifacts_preserves_skipped_exclusion_from_means(tmp_path) -> None:
    result = _evaluation_result()

    write_evaluation_artifacts(result, tmp_path)

    rows = list(csv.DictReader((tmp_path / "summary.csv").open("r", encoding="utf-8")))
    recovery = next(row for row in rows if row["scope"] == "candidate" and row["metric"] == "sequence_recovery")
    assert recovery["count"] == "2"
    assert recovery["ok"] == "1"
    assert recovery["skipped"] == "1"
    assert float(recovery["mean"]) == pytest.approx(0.75)

    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "| `sequence_recovery` | 1/2 | 0.750000 | 0.750000 | ok |" in report
    skipped = (tmp_path / "SKIPPED_METRICS.md").read_text(encoding="utf-8")
    assert "| `sequence_recovery` | skipped | 1 | no_candidate_sequence |" in skipped


def test_write_evaluation_artifacts_writes_flat_metrics_jsonl(tmp_path) -> None:
    result = _evaluation_result()

    write_evaluation_artifacts(result, tmp_path)

    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["scope"] for row in rows} == {"target", "candidate"}
    assert any(row.get("candidate_id") == "target-1:sample-2" and row["status"] == "skipped" for row in rows)


def _evaluation_result() -> dict:
    return {
        "schema_version": "ride_ribodiffusion_eval.v1",
        "protocol_version": "test_protocol",
        "comparability": "unit-test fixture",
        "summary": {
            "target_count": 1,
            "candidate_count": 2,
            "candidate_metrics": {
                "sequence_recovery": {
                    "count": 2,
                    "ok": 1,
                    "skipped": 1,
                    "error": 0,
                    "mean": 0.75,
                    "median": 0.75,
                    "min": 0.75,
                    "max": 0.75,
                }
            },
            "target_metrics": {
                "internal_diversity": {
                    "count": 1,
                    "ok": 1,
                    "skipped": 0,
                    "error": 0,
                    "mean": 0.125,
                    "median": 0.125,
                    "min": 0.125,
                    "max": 0.125,
                }
            },
        },
        "manifest": {"protocol_deviations": ["unit-test deviation"]},
        "targets": [
            {
                "target_id": "target-1",
                "metrics": {
                    "internal_diversity": {
                        "name": "internal_diversity",
                        "status": "ok",
                        "value": 0.125,
                        "reason": None,
                        "details": {},
                        "implementation": "test",
                    }
                },
            }
        ],
        "candidates": [
            {
                "target_id": "target-1",
                "candidate_id": "target-1:sample-1",
                "metrics": {
                    "sequence_recovery": {
                        "name": "sequence_recovery",
                        "status": "ok",
                        "value": 0.75,
                        "reason": None,
                        "details": {},
                        "implementation": "test",
                    }
                },
            },
            {
                "target_id": "target-1",
                "candidate_id": "target-1:sample-2",
                "metrics": {
                    "sequence_recovery": {
                        "name": "sequence_recovery",
                        "status": "skipped",
                        "value": None,
                        "reason": "no_candidate_sequence",
                        "details": {},
                        "implementation": "test",
                    }
                },
            },
        ],
    }
