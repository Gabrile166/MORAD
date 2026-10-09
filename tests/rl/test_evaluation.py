from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.rl.evaluation import summarize_evaluation_results, summarize_target_evaluation


@dataclass(frozen=True)
class Sample:
    tokens: str


@dataclass(frozen=True)
class Record:
    status: str = "ok"
    is_good: bool = False


def test_target_evaluation_summary_tracks_recovery_and_diversity() -> None:
    target = SimpleNamespace(
        target_id="t0",
        sequence_native="AUGCU",
        length=5,
        split="train",
        raw_record={"family": "tRNA"},
    )
    rollout_samples = [Sample("AUGCU"), Sample("AUGCA"), Sample("AUGCA")]
    reward_summary = {
        "fold_cache_hits": 1,
        "fold_cache_misses": 2,
        "metric_cache_hits": 1,
        "metric_cache_misses": 1,
        "tm_score_mean": 0.7,
        "rmsd_mean": 1.2,
        "gdt_ts_mean": 0.6,
    }
    reward_records = [Record(is_good=True), Record(is_good=False), Record(is_good=False)]

    summary = summarize_target_evaluation(
        target=target,
        rollout_samples=rollout_samples,
        reward_summary=reward_summary,
        reward_records=reward_records,
        oracle_calls=2,
    )

    assert summary["sequence_recovery_mean"] == pytest.approx((1.0 + 0.8 + 0.8) / 3.0)
    assert summary["unique_sequence_ratio"] == pytest.approx(2 / 3)
    assert summary["sequence_exact_match_rate"] == pytest.approx(1 / 3)
    assert summary["length_bucket"] == "short"
    assert summary["rna_type"] == "tRNA"
    assert summary["tm_score_mean"] == pytest.approx(0.7)
    assert summary["good_rate"] == pytest.approx(1 / 3)


def test_evaluation_summary_groups_length_buckets() -> None:
    short = {
        "target_id": "s",
        "length_bucket": "short",
        "sequence_recovery_mean": 0.8,
        "sequence_recovery_best": 0.9,
        "sequence_exact_match_rate": 0.1,
        "unique_sequence_ratio": 0.5,
        "duplicate_ratio": 0.5,
        "good_rate": 0.4,
        "valid_rate": 1.0,
        "rmsd_mean": 1.1,
        "tm_score_mean": 0.7,
        "gdt_ts_mean": 0.6,
        "oracle_cache_hit_rate": 0.5,
    }
    long = dict(short, target_id="l", length_bucket="long", sequence_recovery_mean=0.6, unique_sequence_ratio=0.75)

    summary = summarize_evaluation_results(
        [short, long],
        outer_step=2,
        target_cursor=2,
        oracle_calls=3,
        skipped_targets=1,
    )

    assert summary["mode"] == "eval"
    assert summary["targets"] == 2
    assert summary["oracle_calls"] == 3
    assert summary["paper_summary"]["sequence_recovery_mean"] == pytest.approx(0.7)
    assert summary["paper_summary"]["length_buckets"]["short"]["count"] == 1
    assert summary["paper_summary"]["length_buckets"]["long"]["count"] == 1
