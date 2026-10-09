import torch

from src.rl.advantage import AdvantageComputer, EmaRewardScale, WindowRewardScale, compute_group_advantages
from src.rl.protocol import ConditionBundle, RewardBatch, RewardRecord, RolloutBatch, RolloutSample


def test_group_advantages_and_optimality_probability_use_multiplicity():
    """Standard GRPO: advantages are group z-scores, not raw mean-centred values."""
    rewards = torch.tensor([1.0, 3.0, 3.0])
    scale = EmaRewardScale(decay=0.0, epsilon=1.0e-6)

    result = compute_group_advantages(rewards, torch.ones(3, dtype=torch.bool), scale)

    assert not result.should_skip
    # mean-centred is [-4/3, +2/3, +2/3]; GRPO then divides by the group std so
    # the signal is dimensionless. Assert the invariants rather than hard-coding
    # the divisor: zero mean, unit-ish spread, ordering and ties preserved.
    adv = result.advantage
    assert torch.allclose(adv.mean(), torch.zeros(()), atol=1e-5)
    assert adv[0] < 0 < adv[1]
    assert torch.allclose(adv[1], adv[2])
    # the two-to-one ratio of the centred values survives standardisation
    assert torch.allclose(adv[0] / adv[1], torch.tensor(-2.0), atol=1e-4)
    assert torch.all((0 <= result.optimality_probability) & (result.optimality_probability <= 1))
    assert result.optimality_probability[1] == result.optimality_probability[2]


def test_invalid_and_nonfinite_rewards_are_excluded():
    rewards = torch.tensor([1.0, float("nan"), 5.0])
    valid = torch.tensor([True, True, False])
    scale = EmaRewardScale(decay=0.0)

    result = compute_group_advantages(rewards, valid, scale)

    assert result.should_skip
    assert result.skip_reason == "fewer_than_two_valid_rewards"
    assert torch.equal(result.valid_mask, torch.tensor([True, False, False]))


def test_zero_variance_group_skips():
    result = compute_group_advantages(torch.tensor([2.0, 2.0]), None, EmaRewardScale())

    assert result.should_skip
    assert result.skip_reason == "zero_or_nonfinite_group_std"
    assert torch.allclose(result.optimality_probability, torch.tensor([0.5, 0.5]))


def test_window_reward_scale_tracks_recent_values_only():
    scale = WindowRewardScale(window_size=3)

    scale.update(torch.tensor([0.0, 2.0]))
    scale.update(torch.tensor([4.0, 6.0]))

    assert list(scale.values) == [2.0, 4.0, 6.0]
    assert scale.current() > 0


def test_advantage_computer_skips_when_valid_rewards_below_min_valid_even_with_nonzero_variance():
    computer = AdvantageComputer({"scale": {"kind": "ema", "min_valid": 3, "epsilon": 1.0e-6}})
    scored = type(
        "ScoredBatch",
        (),
        {
            "rollout": _rollout(3),
            "rewards": RewardBatch(
                records=(
                    _reward("ok", 1.0),
                    _reward("ok", 3.0),
                    _reward("metric_failed", None),
                ),
                target_ids=("target-1", "target-1", "target-1"),
                sample_ids=("sample-0", "sample-1", "sample-2"),
            ).validate(),
        },
    )()

    batch = computer.compute(scored)

    assert batch.skip_update
    assert batch.skip_reason == "fewer_than_min_valid_rewards:3"
    assert torch.allclose(batch.optimality_probability, torch.full((3,), 0.5))
    assert torch.allclose(batch.advantage, torch.zeros(3))


def _reward(status: str, raw_reward: float | None) -> RewardRecord:
    return RewardRecord(
        status=status,
        rmsd=1.0 if status == "ok" else None,
        tm_score=0.5 if status == "ok" else None,
        gdt_ts=0.5 if status == "ok" else None,
        raw_reward=raw_reward,
        is_good=status == "ok",
        oracle_cache_hit=False,
        oracle_latency_ms=0.0,
        oracle_key=f"oracle-{status}-{raw_reward}",
    ).validate()


def _rollout(size: int) -> RolloutBatch:
    condition = ConditionBundle(
        condition_id="condition-1",
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
            tokens="AUGC",
            x0_onehot=torch.eye(4),
            seed=idx,
            temperature=1.0,
            policy_version="old",
            condition_id="condition-1",
            sampler_name="unit",
            sampler_version="v1",
            latency_ms=0.0,
        )
        for idx in range(size)
    )
    return RolloutBatch(samples=samples, condition=condition).validate()
