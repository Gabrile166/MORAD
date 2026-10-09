import torch

from src.rl.config import ResidualNormalization, TimestepWeighting
from src.rl.losses import (
    DiffusionNFTLossConfig,
    compute_nft_epsilon_loss,
    forward_noise_x0,
    sample_stratified_log_snr_timesteps,
    timestep_weights,
)


def test_stratified_log_snr_timesteps_are_in_open_unit_interval():
    generator = torch.Generator().manual_seed(0)

    t = sample_stratified_log_snr_timesteps(3, 4, -6.0, 6.0, generator=generator)

    assert t.shape == (3, 4)
    assert torch.all((0 < t) & (t < 1))


def test_timestep_weighting_modes_are_finite_and_capped():
    timesteps = torch.tensor([[0.1, 0.5, 0.99]])

    uniform = timestep_weights(timesteps, TimestepWeighting.UNIFORM_EPS)
    velocity = timestep_weights(timesteps, TimestepWeighting.VELOCITY_EQUIVALENT)
    x0 = timestep_weights(timesteps, TimestepWeighting.X0_RECONSTRUCTION, x0_weight_max=5.0)

    assert torch.allclose(uniform, torch.ones_like(timesteps))
    assert torch.all(velocity >= 0)
    assert torch.max(x0) <= 5.0


def test_forward_noise_preserves_shape():
    x0 = torch.zeros(2, 3, 4, 4)
    noise = torch.ones_like(x0)
    timesteps = torch.full((2, 3), 0.5)

    noised = forward_noise_x0(x0, timesteps, noise)

    assert noised.shape == x0.shape
    assert torch.all(noised > 0)


def test_nft_loss_supports_beta_normalization_and_reference_grad():
    current = torch.zeros(2, 3, 4, 4, requires_grad=True)
    old = torch.zeros_like(current)
    target = torch.ones_like(current)
    reference = torch.full_like(current, 0.25)
    r = torch.tensor([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
    timesteps = torch.full((2, 3), 0.5)

    result = compute_nft_epsilon_loss(
        current,
        old,
        target,
        r,
        timesteps,
        epsilon_reference=reference,
        config=DiffusionNFTLossConfig(
            nft_beta=0.2,
            reference_weight=0.1,
            timestep_weighting=TimestepWeighting.UNIFORM_EPS,
            residual_normalization=ResidualNormalization.NONE,
            beta_normalization=True,
        ),
    )
    result.total.backward()

    assert torch.isfinite(result.total)
    assert torch.isfinite(current.grad).all()
    assert result.reference > 0


def test_mean_abs_detached_residual_scale_does_not_require_grad():
    current = torch.zeros(1, 2, 3, 4, requires_grad=True)
    old = torch.zeros_like(current)
    target = torch.ones_like(current)

    result = compute_nft_epsilon_loss(
        current,
        old,
        target,
        torch.full((1, 2), 0.5),
        torch.full((1, 2), 0.5),
        config=DiffusionNFTLossConfig(residual_normalization=ResidualNormalization.MEAN_ABS_DETACHED),
    )

    assert not result.residual_scale_positive.requires_grad
    assert result.residual_scale_positive.shape == (1, 2, 1, 1)
