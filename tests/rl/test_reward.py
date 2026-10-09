from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.rl.protocol import ConditionBundle, RolloutBatch, RolloutSample, TargetRecord
from src.rl.reward import (
    FoldResult,
    JsonDiskCache,
    MetricsOracleProtocol,
    RewardComposer,
    RewardScorer,
    StructuralMetrics,
    compute_structural_metrics,
    fold_cache_key,
    metric_cache_key,
    score_rollout_batch,
)


def test_initial_plan_reward_is_monotonic_across_thresholds() -> None:
    composer = RewardComposer("initial_plan_strict")
    last = None
    for gdt in torch.linspace(0.0, 1.0, 101):
        reward = composer.raw_reward(StructuralMetrics(rmsd=1.0, tm_score=0.5, gdt_ts=float(gdt)))
        if last is not None:
            assert reward >= last
        last = reward

    last = None
    for rmsd in reversed(torch.linspace(0.0, 5.0, 101)):
        reward = composer.raw_reward(StructuralMetrics(rmsd=float(rmsd), tm_score=0.5, gdt_ts=0.5))
        if last is not None:
            assert reward >= last
        last = reward


def test_current_code_compat_is_bit_exact_but_rejected_for_train() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        RewardComposer("current_code_compat", train_mode=True)

    composer = RewardComposer("current_code_compat", train_mode=False)
    metrics = StructuralMetrics(rmsd=1.0, tm_score=0.5, gdt_ts=0.44)
    base = -(1.0 * 0.5) ** 2 + (0.44 * 5.0) ** 2
    assert composer.raw_reward(metrics) == pytest.approx(base + (3.0 - 1.0) * 20.0)

    metrics = StructuralMetrics(rmsd=1.0, tm_score=0.5, gdt_ts=0.46)
    base = -(1.0 * 0.5) ** 2 + (0.46 * 5.0) ** 2
    assert composer.raw_reward(metrics) == pytest.approx(base + (0.46 - 0.45) * 100.0)


def test_good_requires_all_three_gates() -> None:
    """is_good gates on all three primary terms (gdt>=0.50, tm>=0.50, rmsd<=4.0).

    The historical "2 of 3" rule let a sample count as good while failing the
    metric the reward weighted most, which decoupled is_good from the reward.
    """
    composer = RewardComposer()
    # all three clear their gate
    assert composer.is_good(StructuralMetrics(rmsd=3.0, tm_score=0.55, gdt_ts=0.55))
    # rmsd fails (5.0 > 4.0) -> not good even though gdt/tm pass
    assert not composer.is_good(StructuralMetrics(rmsd=5.0, tm_score=0.5, gdt_ts=0.5))
    # tm fails
    assert not composer.is_good(StructuralMetrics(rmsd=1.0, tm_score=0.1, gdt_ts=0.5))
    # gdt fails
    assert not composer.is_good(StructuralMetrics(rmsd=1.0, tm_score=0.5, gdt_ts=0.1))


def test_cache_keys_exclude_reward_composer() -> None:
    fold_a = fold_cache_key("AUGC", "ckpt", "single_seq")
    fold_b = fold_cache_key("AUGC", "ckpt", "single_seq")
    metric_a = metric_cache_key("target", "predicted")
    metric_b = metric_cache_key("target", "predicted")
    assert fold_a == fold_b
    assert metric_a == metric_b
    assert "initial_plan_strict" not in fold_a
    assert "current_code_compat" not in metric_a


def test_dedup_calls_oracle_once_per_unique_sequence_and_expands_multiplicity(tmp_path) -> None:
    target = _target()
    rollout = _rollout(("AUGC", "AUGC", "GGCA"))
    oracle = FakeOracle()
    composer = RewardComposer()

    rewards = score_rollout_batch(
        rollout,
        target,
        oracle,
        composer,
        fold_cache=JsonDiskCache(tmp_path / "fold"),
        oracle_checkpoint_hash="ckpt",
    )

    assert oracle.calls == ["AUGC", "GGCA"]
    assert len(rewards.records) == 3
    assert rewards.records[0].raw_reward == rewards.records[1].raw_reward
    assert rewards.records[0].oracle_key == rewards.records[1].oracle_key
    assert rewards.records[0].status == "ok"


def test_reward_scorer_oracle_calls_count_fold_cache_misses(tmp_path) -> None:
    target = _target()
    rollout = _rollout(("AUGC", "AUGC", "GGCA"))
    oracle = FakeOracle()
    scorer = RewardScorer({"metric_cache": {"dir": str(tmp_path / "cache")}}, oracle=oracle)

    first = scorer.score(rollout, target)
    second = scorer.score(rollout, target)

    assert first.oracle_calls == 2
    assert first.cache_summary["fold_cache_hits"] == 0
    assert first.cache_summary["fold_cache_misses"] == 2
    assert first.cache_summary["unique_sequences"] == 2
    assert second.oracle_calls == 0
    assert second.cache_summary["fold_cache_hits"] == 2
    assert second.cache_summary["fold_cache_misses"] == 0
    assert second.cache_summary["unique_sequences"] == 2
    assert oracle.calls == ["AUGC", "GGCA"]


def test_cached_metrics_are_recomposed_without_oracle_call(tmp_path) -> None:
    target = _target()
    rollout = _rollout(("AUGC",))
    fold_cache = JsonDiskCache(tmp_path / "fold")
    oracle = FakeOracle(rmsd=1.0, gdt=0.49)

    first = score_rollout_batch(rollout, target, oracle, RewardComposer(), fold_cache=fold_cache, oracle_checkpoint_hash="ckpt")
    second = score_rollout_batch(
        rollout,
        target,
        oracle,
        RewardComposer("current_code_compat", train_mode=False),
        fold_cache=fold_cache,
        oracle_checkpoint_hash="ckpt",
    )

    assert oracle.calls == ["AUGC"]
    assert first.records[0].raw_reward != second.records[0].raw_reward
    assert second.records[0].oracle_cache_hit


def test_fold_cache_is_target_independent_but_metrics_are_target_specific(tmp_path) -> None:
    rollout = _rollout(("AUGC",))
    oracle = FakeOracle()
    fold_cache = JsonDiskCache(tmp_path / "fold")
    metric_cache = JsonDiskCache(tmp_path / "metric")
    target_a = _target(target_id="target-a", structure_hash="target-a", offset=0.0)
    target_b = _target(target_id="target-b", structure_hash="target-b", offset=3.0)

    reward_a = score_rollout_batch(
        rollout,
        target_a,
        oracle,
        RewardComposer(),
        fold_cache=fold_cache,
        metric_cache=metric_cache,
        oracle_checkpoint_hash="ckpt",
    )
    reward_b = score_rollout_batch(
        rollout,
        target_b,
        oracle,
        RewardComposer(),
        fold_cache=fold_cache,
        metric_cache=metric_cache,
        oracle_checkpoint_hash="ckpt",
    )

    assert oracle.calls == ["AUGC"]
    assert reward_a.records[0].status == "ok"
    assert reward_b.records[0].status == "ok"
    assert reward_a.records[0].rmsd != reward_b.records[0].rmsd
    assert len(list((tmp_path / "fold").rglob("*.json"))) == 1
    assert len(list((tmp_path / "metric").rglob("*.json"))) == 2


def test_metric_version_invalidates_metric_cache(tmp_path) -> None:
    target = _target()
    rollout = _rollout(("AUGC",))
    oracle = FakeOracle()
    fold_cache = JsonDiskCache(tmp_path / "fold")
    metric_cache = JsonDiskCache(tmp_path / "metric")

    score_rollout_batch(
        rollout,
        target,
        oracle,
        RewardComposer(),
        fold_cache=fold_cache,
        metric_cache=metric_cache,
        oracle_checkpoint_hash="ckpt",
        metric_version="v1",
    )
    score_rollout_batch(
        rollout,
        target,
        oracle,
        RewardComposer(),
        fold_cache=fold_cache,
        metric_cache=metric_cache,
        oracle_checkpoint_hash="ckpt",
        metric_version="v2",
    )

    assert oracle.calls == ["AUGC"]
    assert len(list((tmp_path / "metric").rglob("*.json"))) == 2


def test_real_reward_scorer_requires_existing_checkpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="ckpt is required"):
        RewardScorer({"oracle": {"fake": False}, "metric_cache": {"dir": str(tmp_path / "cache")}})

    with pytest.raises(FileNotFoundError):
        RewardScorer({"oracle": {"fake": False, "ckpt": str(tmp_path / "missing.pt")}, "metric_cache": {"dir": str(tmp_path / "cache")}})


def test_reward_scorer_computes_real_checkpoint_sha(tmp_path) -> None:
    ckpt = tmp_path / "rhofold.pt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    scorer = RewardScorer({"oracle": {"fake": False, "ckpt": str(ckpt)}, "metric_cache": {"dir": str(tmp_path / "cache")}}, oracle=FakeOracle())

    assert scorer.oracle_checkpoint_hash != "unknown"
    assert scorer.manifest_state()["oracle_checkpoint_sha256"] == scorer.oracle_checkpoint_hash


def test_kabsch_metrics_are_rotation_and_translation_invariant() -> None:
    coords = torch.randn(24, 3, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    moved = coords @ q.T + torch.tensor([10.0, -4.0, 2.0], dtype=torch.float64)

    metrics = compute_structural_metrics(moved, coords)

    assert metrics.rmsd < 1.0e-10
    assert metrics.tm_score == pytest.approx(1.0)
    assert metrics.gdt_ts == pytest.approx(1.0)


def test_tm_score_preserves_current_ride_semantics_for_eligible_lengths() -> None:
    coords = torch.randn(24, 3, dtype=torch.float64)
    metrics = compute_structural_metrics(coords, coords)
    assert metrics.tm_score == pytest.approx(1.0)

    short = compute_structural_metrics(coords[:4], coords[:4])
    assert short.tm_score == pytest.approx(0.0)


class FakeOracle(MetricsOracleProtocol):
    def __init__(self, rmsd: float = 1.0, gdt: float = 0.5) -> None:
        self.calls: list[str] = []
        self.rmsd = rmsd
        self.gdt = gdt

    def fold(self, sequence, output_dir, target_id=None):
        self.calls.append(sequence)
        path = output_dir / "pred.pdb"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_pdb(path, _coords())
        return FoldResult(status="ok", predicted_structure_path=str(path), predicted_structure_hash="pred-hash", plddt=0.7, latency_ms=1.0)

    def score_sequence(self, sequence, target, composer, oracle_key, metric_cache=None):
        self.calls.append(sequence)
        metrics = StructuralMetrics(rmsd=self.rmsd, tm_score=0.5, gdt_ts=self.gdt, plddt=0.7)
        return composer.record(metrics, oracle_key=oracle_key, oracle_latency_ms=1.0)


def _target(target_id: str = "target-1", structure_hash: str = "target-hash", offset: float = 0.0) -> TargetRecord:
    return TargetRecord(
        target_id=target_id,
        sequence_native="AUGC",
        length=4,
        structure_ref="target.pdb",
        structure_hash=structure_hash,
        split="train",
        source="unit",
        dataset_version="unit",
        native_oracle_metrics={},
        frozen_ride_metrics={},
        calibration_status="ok",
        reference_c4p_coords=_coords(offset),
    )


def _rollout(sequences: tuple[str, ...]) -> RolloutBatch:
    condition = ConditionBundle(
        condition_id="cond-1",
        target_id="target-1",
        round_id=0,
        node_features=torch.zeros(4, 2),
        edge_features=torch.zeros(0, 2),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        node_mask=torch.ones(4, dtype=torch.bool),
        condition_noise_scale=0.0,
        feature_hash="feature-hash",
    )
    samples = tuple(
        RolloutSample(
            sample_id=f"sample-{idx}",
            target_id="target-1",
            round_id=0,
            tokens=sequence,
            x0_onehot=_onehot(sequence),
            seed=idx,
            temperature=1.0,
            policy_version="old",
            condition_id="cond-1",
            sampler_name="unit",
            sampler_version="v1",
            latency_ms=0.0,
        )
        for idx, sequence in enumerate(sequences)
    )
    return RolloutBatch(samples=samples, condition=condition)


def _onehot(sequence: str) -> torch.Tensor:
    table = {"A": 0, "U": 1, "G": 2, "C": 3}
    tensor = torch.zeros(len(sequence), 4)
    for idx, base in enumerate(sequence):
        tensor[idx, table[base]] = 1.0
    return tensor


def _coords(offset: float = 0.0) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0 + offset, 0.0], [3.0, 1.0, 1.0]],
        dtype=torch.float64,
    )


def _write_pdb(path, coords: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, (x, y, z) in enumerate(coords.tolist(), start=1):
            handle.write(
                f"ATOM  {idx:5d}  C4'   A A{idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
