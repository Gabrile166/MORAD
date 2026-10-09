"""DiffusionNFT epsilon-space utilities for MORAD."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ResidualNormalization, TimestepWeighting


@dataclass(frozen=True)
class DiffusionNFTLossConfig:
    nft_beta: float = 0.1
    reference_weight: float = 0.0
    timestep_weighting: TimestepWeighting = TimestepWeighting.UNIFORM_EPS
    residual_normalization: ResidualNormalization = ResidualNormalization.NONE
    beta_normalization: bool = True
    x0_weight_max: float = 100.0


@dataclass(frozen=True)
class DiffusionNFTLossResult:
    total: torch.Tensor
    policy: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    reference: torch.Tensor
    weights: torch.Tensor
    residual_scale_positive: torch.Tensor
    residual_scale_negative: torch.Tensor


def sample_stratified_log_snr_timesteps(
    batch_size: int,
    draws_per_sample: int,
    log_snr_min: float,
    log_snr_max: float,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if draws_per_sample < 1:
        raise ValueError("draws_per_sample must be >= 1")
    if log_snr_min >= log_snr_max:
        raise ValueError("log_snr_min must be < log_snr_max")
    edges = torch.linspace(log_snr_min, log_snr_max, draws_per_sample + 1, device=device)
    lower = edges[:-1].view(1, draws_per_sample)
    width = (edges[1:] - edges[:-1]).view(1, draws_per_sample)
    u = torch.rand(batch_size, draws_per_sample, device=device, generator=generator)
    log_snr = lower + width * u
    return torch.sigmoid(-log_snr)


def alpha_sigma_from_timesteps(timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.all((0 < timesteps) & (timesteps < 1)):
        raise ValueError("timesteps must be in the open interval (0, 1)")
    log_snr = torch.logit(1.0 - timesteps)
    alpha2 = torch.sigmoid(log_snr)
    sigma2 = torch.sigmoid(-log_snr)
    return torch.sqrt(alpha2), torch.sqrt(sigma2)


def timestep_weights(
    timesteps: torch.Tensor,
    weighting: TimestepWeighting | str,
    *,
    x0_weight_max: float = 100.0,
) -> torch.Tensor:
    weighting = TimestepWeighting(weighting)
    alpha, sigma = alpha_sigma_from_timesteps(timesteps)
    if weighting is TimestepWeighting.UNIFORM_EPS:
        return torch.ones_like(timesteps)
    if weighting is TimestepWeighting.VELOCITY_EQUIVALENT:
        return torch.square(sigma / alpha)
    if x0_weight_max <= 0:
        raise ValueError("x0_weight_max must be > 0")
    return torch.clamp(torch.square(sigma / alpha), max=x0_weight_max)


def forward_noise_x0(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    if x0.shape != noise.shape:
        raise ValueError("x0 and noise shapes must match")
    expanded_t = _expand_time(timesteps, x0.ndim)
    alpha, sigma = alpha_sigma_from_timesteps(expanded_t)
    return alpha * x0 + sigma * noise


def compute_nft_epsilon_loss(
    epsilon_current: torch.Tensor,
    epsilon_old: torch.Tensor,
    epsilon_target: torch.Tensor,
    optimality_probability: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    epsilon_reference: torch.Tensor | None = None,
    config: DiffusionNFTLossConfig | None = None,
    valid_mask: torch.Tensor | None = None,
) -> DiffusionNFTLossResult:
    cfg = config or DiffusionNFTLossConfig()
    if cfg.nft_beta <= 0:
        raise ValueError("nft_beta must be > 0")
    for name, tensor in (
        ("epsilon_old", epsilon_old),
        ("epsilon_target", epsilon_target),
    ):
        if tensor.shape != epsilon_current.shape:
            raise ValueError(f"{name} shape must match epsilon_current")
    if optimality_probability.shape != epsilon_current.shape[:2]:
        raise ValueError("optimality_probability must have shape [B, K]")
    if timesteps.shape != epsilon_current.shape[:2]:
        raise ValueError("timesteps must have shape [B, K]")
    if valid_mask is None:
        valid_mask = torch.ones_like(timesteps, dtype=torch.bool)
    if valid_mask.shape != timesteps.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with shape [B, K]")

    epsilon_old = epsilon_old.detach()
    epsilon_pos = (1.0 - cfg.nft_beta) * epsilon_old + cfg.nft_beta * epsilon_current
    epsilon_neg = (1.0 + cfg.nft_beta) * epsilon_old - cfg.nft_beta * epsilon_current

    weight = timestep_weights(timesteps, cfg.timestep_weighting, x0_weight_max=cfg.x0_weight_max)
    weight_expanded = _expand_like(weight, epsilon_current)
    delta_pos = epsilon_pos - epsilon_target
    delta_neg = epsilon_neg - epsilon_target
    residual_pos = torch.square(delta_pos)
    residual_neg = torch.square(delta_neg)
    scale_pos = _residual_scale(delta_pos, cfg.residual_normalization)
    scale_neg = _residual_scale(delta_neg, cfg.residual_normalization)

    loss_pos_per = _reduce_event_dims(weight_expanded * residual_pos / scale_pos)
    loss_neg_per = _reduce_event_dims(weight_expanded * residual_neg / scale_neg)
    r = optimality_probability
    policy_per = r * loss_pos_per + (1.0 - r) * loss_neg_per
    if cfg.beta_normalization:
        policy_per = policy_per / cfg.nft_beta
    policy = _masked_mean(policy_per, valid_mask)
    positive = _masked_mean(loss_pos_per, valid_mask)
    negative = _masked_mean(loss_neg_per, valid_mask)

    if epsilon_reference is None or cfg.reference_weight == 0:
        reference = torch.zeros((), dtype=epsilon_current.dtype, device=epsilon_current.device)
    else:
        if epsilon_reference.shape != epsilon_current.shape:
            raise ValueError("epsilon_reference shape must match epsilon_current")
        reference_per = _reduce_event_dims(torch.square(epsilon_current - epsilon_reference.detach()))
        reference = _masked_mean(reference_per, valid_mask)
    total = policy + cfg.reference_weight * reference
    return DiffusionNFTLossResult(
        total=total,
        policy=policy,
        positive=positive,
        negative=negative,
        reference=reference,
        weights=weight,
        residual_scale_positive=scale_pos.detach(),
        residual_scale_negative=scale_neg.detach(),
    )


def _expand_time(timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
    result = timesteps
    while result.ndim < ndim:
        result = result.unsqueeze(-1)
    return result


def _expand_like(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _expand_time(values, target.ndim)


def _reduce_event_dims(value: torch.Tensor) -> torch.Tensor:
    if value.ndim <= 2:
        return value
    dims = tuple(range(2, value.ndim))
    return value.mean(dim=dims)


def _residual_scale(residual: torch.Tensor, mode: ResidualNormalization | str) -> torch.Tensor:
    mode = ResidualNormalization(mode)
    if mode is ResidualNormalization.NONE:
        return torch.ones((), dtype=residual.dtype, device=residual.device)
    dims = tuple(range(2, residual.ndim))
    if not dims:
        scale = residual.detach().abs().mean(dim=-1, keepdim=True)
    else:
        scale = residual.detach().abs().mean(dim=dims, keepdim=True)
    return torch.clamp(scale, min=1.0e-8)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        # All samples invalid in this microbatch: return zero loss with correct
        # device/dtype so DDP all-reduce still succeeds (gradients are zero).
        return values.sum() * 0.0
    return selected.mean()
