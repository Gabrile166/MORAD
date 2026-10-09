import pytest
import torch

from src.rl.protocol import (
    ConditionBundle,
    RewardBatch,
    RewardRecord,
    RolloutBatch,
    RolloutSample,
    TargetRecord,
    TrainBatch,
    train_batch_from_rollout,
)


def _onehot(tokens=(0, 1, 2)):
    return torch.nn.functional.one_hot(torch.tensor(tokens), num_classes=4).float()


def _condition():
    return ConditionBundle(
        condition_id="cond-1",
        target_id="target-1",
        round_id=0,
        node_features=torch.zeros(3, 5),
        edge_features=torch.zeros(2, 7),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        node_mask=torch.ones(3, dtype=torch.bool),
        condition_noise_scale=0.0,
        feature_hash="hash",
    )


def test_target_and_condition_validate():
    TargetRecord(
        target_id="target-1",
        sequence_native="ACG",
        length=3,
        structure_ref="native.cif",
        structure_hash="hash",
        split="train",
        source="fixture",
        dataset_version="v0",
        native_oracle_metrics={},
        frozen_ride_metrics={},
        calibration_status="ok",
    ).validate()
    _condition().validate()


def test_rollout_batch_select_and_cpu():
    condition = _condition()
    samples = tuple(
        RolloutSample(
            sample_id=f"s-{idx}",
            target_id="target-1",
            round_id=0,
            tokens="ACG",
            x0_onehot=_onehot(),
            seed=idx,
            temperature=1.0,
            policy_version="old-1",
            condition_id="cond-1",
            sampler_name="adaptive_x0_renoise",
            sampler_version="v1",
            latency_ms=1.0,
        )
        for idx in range(2)
    )
    batch = RolloutBatch(samples=samples, condition=condition).validate()

    selected = batch.select([True, False]).cpu().validate()

    assert len(selected.samples) == 1
    assert selected.samples[0].sample_id == "s-0"
    assert selected.samples[0].x0_onehot.device.type == "cpu"


def test_onehot_validation_rejects_continuous_latent():
    bad = _onehot()
    bad[0, 0] = 0.5
    sample = RolloutSample(
        sample_id="s",
        target_id="target-1",
        round_id=0,
        tokens="ACG",
        x0_onehot=bad,
        seed=1,
        temperature=1.0,
        policy_version="old",
        condition_id="cond-1",
        sampler_name="adaptive_x0_renoise",
        sampler_version="v1",
        latency_ms=0.0,
    )

    with pytest.raises(ValueError, match="0 or 1"):
        sample.validate()


def test_train_batch_validate_select_and_builder():
    condition = _condition()
    samples = tuple(
        RolloutSample(
            sample_id=f"s-{idx}",
            target_id="target-1",
            round_id=0,
            tokens="ACG",
            x0_onehot=_onehot(),
            seed=idx,
            temperature=1.0,
            policy_version="old",
            condition_id="cond-1",
            sampler_name="adaptive_x0_renoise",
            sampler_version="v1",
            latency_ms=0.0,
        )
        for idx in range(2)
    )
    rollout = RolloutBatch(samples=samples, condition=condition)
    rewards = RewardBatch(
        records=(
            RewardRecord("ok", 1.0, 0.5, 0.6, 10.0, True, False, 1.0, "k1"),
            RewardRecord("ok", 2.0, 0.4, 0.4, 2.0, False, True, 0.0, "k2"),
        ),
        target_ids=("target-1", "target-1"),
        sample_ids=("s-0", "s-1"),
    )
    batch = train_batch_from_rollout(
        rollout,
        rewards,
        optimality_probability=torch.tensor([1.0, 0.0]),
        advantage=torch.tensor([4.0, -4.0]),
    )

    batch.validate()
    selected = batch.select(torch.tensor([False, True])).validate()

    assert selected.sample_ids == ("s-1",)
    assert selected.x0_onehot.shape == (1, 3, 4)


def test_train_batch_condition_id_mismatch_fails():
    batch = TrainBatch(
        x0_onehot=_onehot().unsqueeze(0),
        condition=_condition(),
        optimality_probability=torch.tensor([0.5]),
        raw_reward=torch.tensor([1.0]),
        advantage=torch.tensor([0.0]),
        target_ids=("target-1",),
        sample_ids=("s",),
        policy_versions=("old",),
        condition_ids=("wrong",),
        valid_mask=torch.ones(1, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="condition_ids"):
        batch.validate()
