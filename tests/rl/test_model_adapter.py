from types import SimpleNamespace

import torch

from src.rl.model_adapter import RIDEModelAdapter


class CountingPolicy(torch.nn.Module):
    out_dim = 4

    def __init__(self):
        super().__init__()
        self.encode_calls = 0

    def encode_condition(self, condition):
        self.encode_calls += 1
        return condition.seq.float().sum()

    def predict_noise_from_encoding(self, encoding, z_t, noise_level, time=None):
        return z_t * 0.0 + encoding.custom.to(z_t.device) * 0.01 + noise_level.view(-1, 1, 1) * 0.0


class ForwardOnlyPolicy(torch.nn.Module):
    out_dim = 4

    def forward(self, batch, time=None, noise_level=None):
        return (batch.z_t * 0.25).unsqueeze(0)


def condition(condition_id="cond0"):
    return SimpleNamespace(condition_id=condition_id, seq=torch.arange(5))


def test_old_policy_encoding_cache_is_keyed_by_condition_and_version():
    policy = CountingPolicy()
    adapter = RIDEModelAdapter(policy)
    first = adapter.encode_condition(condition("a"), "old", "v1")
    second = adapter.encode_condition(condition("a"), "old", "v1")
    third = adapter.encode_condition(condition("a"), "old", "v2")

    assert first is second
    assert third is not first
    assert policy.encode_calls == 2


def test_current_policy_cache_lives_only_inside_graph_context():
    policy = CountingPolicy()
    adapter = RIDEModelAdapter(policy)
    cond = condition("a")

    with adapter.current_graph_cache("graph-a"):
        first = adapter.encode_condition(cond, "current", "v1")
        second = adapter.encode_condition(cond, "current", "v1")
    with adapter.current_graph_cache("graph-b"):
        third = adapter.encode_condition(cond, "current", "v1")

    assert first is second
    assert third is not first
    assert policy.encode_calls == 2


def test_predict_noise_supports_batched_latents():
    policy = CountingPolicy()
    adapter = RIDEModelAdapter(policy)
    enc = adapter.encode_condition(condition(), "old", "v1")
    z_t = torch.randn(3, 5, 4)
    noise = adapter.predict_noise(enc, z_t, torch.tensor([0.1, 0.2, 0.3]))

    assert noise.shape == z_t.shape
    assert torch.allclose(noise[0], torch.full((5, 4), 0.1))


def test_forward_fallback_does_not_mutate_condition_z_t():
    policy = ForwardOnlyPolicy()
    adapter = RIDEModelAdapter(policy)
    cond = condition()
    cond.z_t = torch.full((5, 4), -99.0)
    enc = adapter.encode_condition(cond, "old", "v1")
    z_t = torch.ones(5, 4)

    noise = adapter.predict_noise(enc, z_t, torch.tensor(0.5))

    assert torch.allclose(noise, torch.full((5, 4), 0.25))
    assert torch.all(cond.z_t == -99.0)
