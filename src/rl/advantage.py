"""Reward scaling, group advantages, and optimality probabilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from src.rl.protocol import RewardBatch, TrainBatch, train_batch_from_rollout


@dataclass(frozen=True)
class AdvantageResult:
    advantage: torch.Tensor
    optimality_probability: torch.Tensor
    scale: float
    group_mean: float
    group_std: float
    valid_mask: torch.Tensor
    should_skip: bool
    skip_reason: str | None = None


class EmaRewardScale:
    """Recent reward std estimate stored as checkpointable scalar state."""

    def __init__(self, decay: float = 0.95, epsilon: float = 1.0e-6, warmup_count: int = 1) -> None:
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        self.decay = decay
        self.epsilon = epsilon
        self.warmup_count = warmup_count
        self.count = 0
        self.value = 0.0

    def update(self, rewards: torch.Tensor, valid_mask: torch.Tensor | None = None) -> float:
        current = _finite_std(rewards, valid_mask)
        if current is None:
            return self.current()
        self.count += 1
        if self.count <= self.warmup_count or self.value <= 0:
            self.value = current
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * current
        return self.current()

    def current(self) -> float:
        return max(float(self.value), self.epsilon)

    def state_dict(self) -> dict[str, float | int]:
        return {"count": self.count, "value": self.value, "decay": self.decay, "epsilon": self.epsilon}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.count = int(state["count"])
        self.value = float(state["value"])


class WindowRewardScale:
    """Finite-window recent reward std estimate."""

    def __init__(self, window_size: int = 128, epsilon: float = 1.0e-6) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        self.window_size = window_size
        self.epsilon = epsilon
        self.values: deque[float] = deque(maxlen=window_size)

    def update(self, rewards: torch.Tensor, valid_mask: torch.Tensor | None = None) -> float:
        selected = _select_finite(rewards, valid_mask)
        self.values.extend(float(item) for item in selected.cpu())
        return self.current()

    def current(self) -> float:
        if len(self.values) < 2:
            return self.epsilon
        tensor = torch.tensor(list(self.values), dtype=torch.float32)
        return max(float(tensor.std(unbiased=False).item()), self.epsilon)

    def state_dict(self) -> dict[str, object]:
        return {"values": list(self.values), "window_size": self.window_size, "epsilon": self.epsilon}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.values.clear()
        self.values.extend(float(item) for item in state["values"])  # type: ignore[index]


def compute_group_advantages(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor | None,
    reward_scale: EmaRewardScale | WindowRewardScale,
    update_scale: bool = True,
    group_normalize: bool = True,
    z_clip: float = 2.0,
    group_stats: tuple[float, float] | None = None,
) -> AdvantageResult:
    """Group-relative advantages.

    With ``group_normalize=True`` this is textbook GRPO: subtract the group mean
    and divide by the group standard deviation, giving a dimensionless z-score.
    ``z_clip`` bounds it so one outlier cannot dominate the update.

    ``group_stats`` lets a caller supply ``(mean, std)`` computed over a *larger*
    set than the local tensor. That is required when a single target's rollout
    group is split across several ranks: each rank holds only part of the group,
    so normalising against local statistics would produce a different, biased
    baseline per rank and break the group-relative objective. The ranks sharing a
    target exchange their rewards first and pass the merged statistics here.
    """
    if rewards.ndim != 1:
        raise ValueError("rewards must have shape [G]")
    if valid_mask is None:
        valid_mask = torch.ones_like(rewards, dtype=torch.bool)
    if valid_mask.shape != rewards.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with same shape as rewards")

    finite_valid = valid_mask & torch.isfinite(rewards)
    selected = rewards[finite_valid]
    zeros = torch.zeros_like(rewards)
    if selected.numel() < 2:
        return AdvantageResult(zeros, torch.full_like(rewards, 0.5), reward_scale.current(), 0.0, 0.0, finite_valid, True, "fewer_than_two_valid_rewards")

    if group_stats is not None:
        merged_mean, merged_std = group_stats
        group_mean_tensor = torch.as_tensor(
            float(merged_mean), dtype=rewards.dtype, device=rewards.device
        )
        group_std = torch.as_tensor(
            float(merged_std), dtype=rewards.dtype, device=rewards.device
        )
    else:
        group_mean_tensor = selected.mean()
        group_std = selected.std(unbiased=False)

    if not torch.isfinite(group_std) or group_std <= 0:
        return AdvantageResult(zeros, torch.full_like(rewards, 0.5), reward_scale.current(), float(group_mean_tensor.item()), 0.0, finite_valid, True, "zero_or_nonfinite_group_std")

    if update_scale:
        scale = reward_scale.update(rewards, finite_valid)
    else:
        scale = reward_scale.current()

    advantage = torch.zeros_like(rewards)
    centered = rewards[finite_valid] - group_mean_tensor
    if group_normalize:
        # Standard GRPO: dimensionless z-score, clipped for stability.
        z = centered / (group_std + 1.0e-8)
        if z_clip > 0:
            z = torch.clamp(z, -float(z_clip), float(z_clip))
        advantage[finite_valid] = z
    else:
        advantage[finite_valid] = centered

    optimality = torch.full_like(rewards, 0.5)
    if group_normalize:
        # The advantage is already normalised, so mapping it through the EMA
        # reward scale (whose magnitude is unrelated to a z-score) would shrink
        # the signal towards 0.5 and starve the gradient. Rescale by z_clip
        # instead, which keeps the full [0, 1] range in play.
        denom = float(z_clip) if z_clip > 0 else 3.0
        optimality[finite_valid] = 0.5 + 0.5 * torch.clamp(
            advantage[finite_valid] / denom, -1.0, 1.0
        )
    else:
        optimality[finite_valid] = 0.5 + 0.5 * torch.clamp(
            advantage[finite_valid] / scale, -1.0, 1.0
        )
    return AdvantageResult(
        advantage=advantage,
        optimality_probability=optimality,
        scale=scale,
        group_mean=float(group_mean_tensor.item()),
        group_std=float(group_std.item()),
        valid_mask=finite_valid,
        should_skip=False,
    )


def _finite_std(rewards: torch.Tensor, valid_mask: torch.Tensor | None) -> float | None:
    selected = _select_finite(rewards, valid_mask)
    if selected.numel() < 2:
        return None
    return float(selected.std(unbiased=False).item())


def _select_finite(rewards: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if rewards.ndim != 1:
        raise ValueError("rewards must have shape [N]")
    mask = torch.isfinite(rewards)
    if valid_mask is not None:
        if valid_mask.shape != rewards.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with same shape as rewards")
        mask = mask & valid_mask
    return rewards[mask]


class AdvantageComputer:
    """Turn a scored rollout group into a TrainBatch for DiffusionNFT updates."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        scale_cfg = dict(cfg.get("scale", {}))
        kind = str(scale_cfg.get("kind", "ema"))
        epsilon = float(scale_cfg.get("epsilon", 1.0e-6))
        if kind == "window":
            self.reward_scale: EmaRewardScale | WindowRewardScale = WindowRewardScale(
                window_size=int(scale_cfg.get("window_size", 128)),
                epsilon=epsilon,
            )
        elif kind == "ema":
            self.reward_scale = EmaRewardScale(
                decay=float(scale_cfg.get("decay", scale_cfg.get("ema_decay", 0.95))),
                epsilon=epsilon,
                warmup_count=int(scale_cfg.get("warmup_count", 1)),
            )
        else:
            raise ValueError(f"unknown advantage scale kind: {kind}")
        self.min_valid = int(scale_cfg.get("min_valid", 2))
        # Standard GRPO by default; both knobs are config-visible so an ablation
        # can turn normalisation off without touching code.
        self.group_normalize = bool(cfg.get("group_normalize", True))
        self.z_clip = float(cfg.get("z_clip", 2.0))

    def compute(self, scored_batch: Any, group_stats: tuple[float, float] | None = None) -> TrainBatch:
        rollout = getattr(scored_batch, "rollout", None)
        rewards = getattr(scored_batch, "rewards", scored_batch)
        if rollout is None:
            raise ValueError("scored_batch must expose rollout and rewards")
        if not isinstance(rewards, RewardBatch):
            raise TypeError("scored_batch.rewards must be a RewardBatch")
        raw = torch.tensor(
            [float(record.raw_reward) if record.raw_reward is not None else float("nan") for record in rewards.records],
            dtype=torch.float32,
        )
        valid = torch.tensor([record.status == "ok" for record in rewards.records], dtype=torch.bool)
        if int(valid.sum().item()) < self.min_valid:
            neutral = torch.full_like(raw, 0.5)
            zeros = torch.zeros_like(raw)
            batch = train_batch_from_rollout(rollout, rewards, neutral, zeros)
            return replace(
                batch,
                skip_update=True,
                skip_reason=f"fewer_than_min_valid_rewards:{self.min_valid}",
            )
        else:
            result = compute_group_advantages(
                raw,
                valid,
                self.reward_scale,
                group_normalize=self.group_normalize,
                z_clip=self.z_clip,
                group_stats=group_stats,
            )
        batch = train_batch_from_rollout(
            rollout,
            rewards,
            result.optimality_probability,
            result.advantage,
        )
        if result.should_skip:
            batch = replace(batch, skip_update=True, skip_reason=result.skip_reason)
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "reward_scale": self.reward_scale.state_dict(),
            "scale_type": type(self.reward_scale).__name__,
            "min_valid": self.min_valid,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "reward_scale" in state:
            self.reward_scale.load_state_dict(dict(state["reward_scale"]))
