from types import SimpleNamespace

import torch

from src.constants import NUM_TO_LETTER
from src.diffusion import ddim_sample_with_logprob
from src.noise_schedule import NoiseScheduleVP
from src.rl.protocol import ConditionBundle as ProtocolConditionBundle
from src.rl.protocol import RolloutBatch as ProtocolRolloutBatch
from src.rl.rollout import (
    AdaptiveX0RenoiseRollout,
    AdaptiveX0RenoiseRolloutConfig,
    RolloutState,
)


class ZeroNoisePolicy(torch.nn.Module):
    out_dim = 4

    def forward(self, batch, time=None, noise_level=None):
        return torch.zeros_like(batch.z_t).unsqueeze(0)


def condition(length=6):
    return SimpleNamespace(
        condition_id="cond0",
        target_id="target0",
        seq=torch.arange(length) % 4,
    )


def protocol_condition(length=6):
    return ProtocolConditionBundle(
        condition_id="cond0",
        target_id="target0",
        round_id=0,
        node_features=torch.zeros(length, 3),
        edge_features=torch.zeros(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        node_mask=torch.ones(length, dtype=torch.bool),
        condition_noise_scale=0.0,
        feature_hash="feature0",
    )


def scheduler():
    return NoiseScheduleVP(schedule="linear", continuous_beta_0=0.1, continuous_beta_1=0.2, eps=1e-3)


def test_rollout_is_fixed_seed_reproducible_and_outputs_onehot():
    policy = ZeroNoisePolicy()
    strategy = AdaptiveX0RenoiseRollout(
        scheduler(),
        AdaptiveX0RenoiseRolloutConfig(group_size=3, n_steps=6, temperature=0.7),
    )
    state = RolloutState(round_id="r0", base_seed=123, policy_version="old-1")

    first = strategy.generate(SimpleNamespace(target_id="target0"), protocol_condition(), policy, state)
    second = strategy.generate(SimpleNamespace(target_id="target0"), protocol_condition(), policy, state)

    assert [sample.seed for sample in first.samples] == [123, 124, 125]
    for left, right in zip(first.samples, second.samples):
        assert left.tokens == right.tokens
        assert torch.equal(left.x0_onehot, right.x0_onehot)
        assert set(left.x0_onehot.unique().tolist()).issubset({0.0, 1.0})
        assert torch.all(left.x0_onehot.sum(dim=-1) == 1)


def test_group_temperature_shared_eval_probe_excluded_from_train_samples():
    strategy = AdaptiveX0RenoiseRollout(
        scheduler(),
        AdaptiveX0RenoiseRolloutConfig(group_size=2, n_steps=5, temperature=0.8),
    )
    batch = strategy.generate(
        SimpleNamespace(target_id="target0"),
        protocol_condition(),
        ZeroNoisePolicy(),
        RolloutState(round_id="r0", base_seed=10, policy_version="old-1"),
    )

    assert len(batch.samples) == 2
    assert {sample.temperature for sample in batch.samples} == {0.8}
    assert strategy.last_eval_probe is not None
    assert strategy.last_eval_probe.temperature == 0.0
    assert strategy.last_eval_probe.sample_id not in {sample.sample_id for sample in batch.samples}


def test_rollout_does_not_mutate_condition_z_t():
    cond = protocol_condition()
    strategy = AdaptiveX0RenoiseRollout(
        scheduler(),
        AdaptiveX0RenoiseRolloutConfig(group_size=1, n_steps=4),
    )

    strategy.generate(
        SimpleNamespace(target_id="target0"),
        cond,
        ZeroNoisePolicy(),
        RolloutState(base_seed=5, policy_version="old-1"),
    )

    assert not hasattr(cond, "z_t")


def test_zero_temperature_matches_existing_x0_renoise_sampler_tokens():
    torch.manual_seed(77)
    cond_for_old = condition(length=5)
    policy = ZeroNoisePolicy()
    noise_scheduler = scheduler()
    old_latent, _, _ = ddim_sample_with_logprob(
        policy,
        noise_scheduler,
        cond_for_old,
        n_steps=6,
        device=torch.device("cpu"),
        temperature=1.0,
        deterministic=True,
    )

    strategy = AdaptiveX0RenoiseRollout(
        noise_scheduler,
        AdaptiveX0RenoiseRolloutConfig(
            group_size=0,
            n_steps=6,
            temperature=1.0,
            include_eval_probe=True,
            keep_final_latent_debug=True,
        ),
    )
    batch = strategy.generate(
        SimpleNamespace(target_id="target0"),
        protocol_condition(length=5),
        policy,
        RolloutState(base_seed=78, policy_version="old-1"),
    )

    expected_tokens = "".join(NUM_TO_LETTER[int(token)] for token in torch.argmax(old_latent, dim=-1).tolist())
    assert strategy.last_eval_probe is not None
    assert strategy.last_eval_probe.tokens == expected_tokens


def test_protocol_condition_returns_protocol_batch_and_keeps_eval_probe_out():
    strategy = AdaptiveX0RenoiseRollout(
        scheduler(),
        AdaptiveX0RenoiseRolloutConfig(group_size=2, n_steps=4, include_eval_probe=True),
    )
    batch = strategy.generate(
        SimpleNamespace(target_id="target0"),
        protocol_condition(),
        ZeroNoisePolicy(),
        RolloutState(round_id=0, base_seed=8, policy_version="old-1"),
    )

    assert isinstance(batch, ProtocolRolloutBatch)
    assert len(batch.samples) == 2
    assert strategy.last_eval_probe is not None
    assert strategy.last_eval_probe.sample_id not in {sample.sample_id for sample in batch.samples}
    batch.validate()


def test_rollout_rescue_state_machine_increases_temperature_on_low_diversity():
    strategy = AdaptiveX0RenoiseRollout(
        scheduler(),
        AdaptiveX0RenoiseRolloutConfig(
            max_rescues=1,
            rescue_min_unique_sequences=2,
            rescue_min_valid_sequences=2,
            temperature=0.3,
            temperature_max=0.6,
            rescue_temperature_step=0.2,
        ),
    )
    samples = [SimpleNamespace(tokens="AUGC"), SimpleNamespace(tokens="AUGC")]
    records = [SimpleNamespace(status="ok"), SimpleNamespace(status="metric_failed")]
    scored = SimpleNamespace(
        rollout=SimpleNamespace(samples=samples),
        rewards=SimpleNamespace(records=records),
    )

    next_state = strategy.next_round(scored, RolloutState(temperature=0.3))

    assert next_state is not None
    assert next_state.rescue_count == 1
    assert next_state.temperature == 0.5
    assert "unique<2" in strategy.state_dict()["last_rescue_reason"]
    assert strategy.next_round(scored, next_state) is None
