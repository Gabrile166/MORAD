"""DiffusionNFT-epsilon trainer integration for RIDER RL."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import torch

from src.rl import distributed as dist_utils

from .config import TrainerConfig, parse_config


LossFn = Callable[..., Mapping[str, torch.Tensor]]


@dataclass
class TrainerState:
    optimizer_step: int = 0
    successful_steps: int = 0
    skipped_steps: int = 0
    accumulation_step: int = 0
    last_loss: float | None = None
    last_grad_norm: float | None = None
    ema_decay: float | None = None


class DiffusionNFTTrainer:
    """Owns optimizer, finite guards, gradient updates, and old-policy EMA."""

    def __init__(
        self,
        *,
        current_policy: torch.nn.Module,
        old_policy: torch.nn.Module,
        reference_policy: torch.nn.Module | None = None,
        loss_fn: LossFn | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        config: TrainerConfig | Mapping[str, Any] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.current_policy = current_policy
        self.old_policy = old_policy
        self.reference_policy = reference_policy
        self.config = self._coerce_config(config)
        self.device = torch.device(device) if device is not None else next(current_policy.parameters()).device
        self.current_policy.to(self.device)
        self.old_policy.to(self.device)
        if self.reference_policy is not None:
            self.reference_policy.to(self.device)
        self.loss_fn = loss_fn or self._compute_default_nft_loss
        self.optimizer = optimizer or torch.optim.AdamW(
            self.current_policy.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.optim.amp and self.device.type == "cuda")
        self.scheduler = self._build_scheduler()
        self.state = TrainerState()
        self._set_policy_modes()

    def _build_scheduler(self) -> Any:
        """Warmup + cosine decay, driven by optimizer updates.

        `lr_total_steps` must be the number of *updates* the run will perform
        (targets / batch_targets * epochs), not the number of targets: the
        schedule advances once per update. A warmup avoids the large early steps
        that a freshly initialised advantage scale would otherwise amplify, and
        the cosine tail lets the policy settle instead of oscillating.
        """
        optim_cfg = self.config.optim
        kind = str(getattr(optim_cfg, "lr_schedule", "constant")).lower()
        if kind in {"", "constant", "none"}:
            return None
        total = int(getattr(optim_cfg, "lr_total_steps", 0) or 0)
        if total <= 0:
            return None
        warmup = max(1, int(total * float(getattr(optim_cfg, "lr_warmup_ratio", 0.05))))
        min_ratio = float(getattr(optim_cfg, "lr_min_ratio", 0.1))

        def lr_lambda(step: int) -> float:
            import math

            if step < warmup:
                # Linear ramp; +1 so the very first update is not exactly zero.
                return float(step + 1) / float(warmup)
            if kind != "warmup_cosine":
                return 1.0
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    @staticmethod
    def _coerce_config(config: TrainerConfig | Mapping[str, Any] | None) -> TrainerConfig:
        if isinstance(config, TrainerConfig):
            return config
        return parse_config({"trainer": dict(config or {})}).trainer

    def _set_policy_modes(self) -> None:
        self.current_policy.eval()
        self.old_policy.eval()
        for parameter in self.old_policy.parameters():
            parameter.requires_grad_(False)
        if self.reference_policy is not None:
            self.reference_policy.eval()
            for parameter in self.reference_policy.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_current_policy(
        cls,
        current_policy: torch.nn.Module,
        *,
        reference_policy: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> "DiffusionNFTTrainer":
        old_policy = copy.deepcopy(current_policy)
        if reference_policy is None:
            reference_policy = copy.deepcopy(current_policy)
        return cls(
            current_policy=current_policy,
            old_policy=old_policy,
            reference_policy=reference_policy,
            **kwargs,
        )

    def train_step(self, train_batch: Any) -> dict[str, Any]:
        """Run one optimizer step from a protocol TrainBatch-like object."""
        self._set_policy_modes()
        if self.state.accumulation_step == 0:
            self.optimizer.zero_grad(set_to_none=True)
        batch_size = self._batch_size(train_batch)
        microbatches = self._iter_microbatches(train_batch, batch_size)
        grad_accum = max(1, int(self.config.optim.gradient_accumulation_steps))
        weighted_losses: dict[str, torch.Tensor] = {}
        # Set when any microbatch produced a non-finite loss. Handled after the
        # loop so the collective schedule stays identical on every rank.
        local_nonfinite_loss = False

        for micro_batch, micro_size in microbatches:
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                losses = self.loss_fn(
                    train_batch=micro_batch,
                    current_policy=self.current_policy,
                    old_policy=self.old_policy,
                    reference_policy=self.reference_policy,
                )
                total_loss = losses["loss_total"]
                scale = float(micro_size) / float(max(1, batch_size))
                scaled_loss = total_loss * (scale / float(grad_accum))
            if not torch.isfinite(total_loss).all():
                # Do NOT all-reduce or return from inside this loop. Ranks split
                # their batch into different numbers of microbatches and hit this
                # branch at different iterations, so collectives issued here are
                # not rank-symmetric and NCCL pairs mismatched operations, which
                # busy-waits forever. Record it and let the single reduce below
                # -- which every rank always reaches -- handle the step.
                local_nonfinite_loss = True
                continue
            self.scaler.scale(scaled_loss).backward()
            for key, value in losses.items():
                if hasattr(value, "detach"):
                    weighted_losses[key] = weighted_losses.get(key, torch.zeros((), device=value.device, dtype=value.dtype)) + value.detach() * scale

        self.state.accumulation_step += 1
        if self.state.accumulation_step < grad_accum:
            loss_value = float(weighted_losses["loss_total"].detach().cpu()) if "loss_total" in weighted_losses else None
            self.state.last_loss = loss_value
            return {
                "status": "accumulating",
                "optimizer_step": self.state.optimizer_step,
                "accumulation_step": self.state.accumulation_step,
                "gradient_accumulation_steps": grad_accum,
                "loss_total": loss_value,
            }

        # Every rank must agree on whether this step is usable *before* anyone
        # branches, otherwise the branches diverge and the collectives desync.
        self.scaler.unscale_(self.optimizer)
        # Average gradients across ranks *before* clipping, so grad_norm and the
        # clip threshold apply to the real batch gradient. This is exactly what
        # DistributedDataParallel does internally; we do it explicitly because
        # each rank traces a differently shaped graph (targets vary in length),
        # which DDP's bucketed autograd hooks cannot handle.
        any_nonfinite = dist_utils.average_gradients(
            self.current_policy.parameters(), nonfinite_flag=local_nonfinite_loss
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.current_policy.parameters(),
            self.config.optim.grad_clip_norm,
            error_if_nonfinite=False,
        )
        if any_nonfinite:
            # Some rank saw a non-finite loss. All ranks have now completed the
            # reduce, so it is safe to drop the step -- symmetrically.
            self.state.skipped_steps += 1
            self.state.accumulation_step = 0
            self.optimizer.zero_grad(set_to_none=True)
            self._step_scheduler()
            return {"status": "skipped_nonfinite_loss", "loss_total": float("nan")}
        if not torch.isfinite(grad_norm):
            self.state.skipped_steps += 1
            self.state.accumulation_step = 0
            self.optimizer.zero_grad(set_to_none=True)
            # Gradients were already reduced above, so no rank is left waiting;
            # keep the LR schedule in lockstep across ranks and move on.
            self._step_scheduler()
            loss_value = float(weighted_losses["loss_total"].detach().cpu()) if "loss_total" in weighted_losses else float("nan")
            return {"status": "skipped_nonfinite_grad", "loss_total": loss_value}

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self._step_scheduler()
        self.state.optimizer_step += 1
        self.state.successful_steps += 1
        self.state.accumulation_step = 0
        self.state.last_loss = float(weighted_losses["loss_total"].detach().cpu()) if "loss_total" in weighted_losses else None
        self.state.last_grad_norm = float(grad_norm.detach().cpu())
        self._maybe_update_old_policy()

        metrics = {
            "status": "ok",
            "optimizer_step": self.state.optimizer_step,
            "accumulation_step": self.state.accumulation_step,
            "gradient_accumulation_steps": grad_accum,
            "loss_total": self.state.last_loss,
            "grad_norm": self.state.last_grad_norm,
            "ema_decay": self.state.ema_decay,
            "lr": self._current_lr(),
        }
        for key, value in weighted_losses.items():
            if hasattr(value, "detach"):
                metrics[key] = float(value.detach().cpu())
        return metrics

    def _current_lr(self) -> float:
        for group in self.optimizer.param_groups:
            return float(group.get("lr", 0.0))
        return 0.0

    def _step_scheduler(self) -> None:
        """Advance the LR schedule, but only on real optimizer updates.

        Kept separate so the several early-return paths can call it without
        duplicating the None check.
        """
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            scheduler.step()

    def train_step_empty(self) -> dict[str, Any]:
        """Participate in the update without contributing a real gradient.

        Needed when a rank's batch is unusable (all-invalid group, degenerate
        advantages). Under DDP the collectives are rank-symmetric, so such a rank
        cannot simply skip: it materialises zero gradients, joins the all-reduce,
        and steps. The other ranks' gradients are unaffected apart from being
        divided by the full world size, which is the correct average over the
        batch that actually produced signal.
        """
        self._set_policy_modes()
        if self.state.accumulation_step == 0:
            self.optimizer.zero_grad(set_to_none=True)
        for param in self.current_policy.parameters():
            if param.requires_grad and param.grad is None:
                param.grad = torch.zeros_like(param)

        grad_accum = max(1, int(self.config.optim.gradient_accumulation_steps))
        self.state.accumulation_step += 1
        if self.state.accumulation_step < grad_accum:
            # Mirrors train_step's accumulating branch: no collectives on either
            # side, so the two stay in step.
            return {
                "status": "accumulating_empty",
                "optimizer_step": self.state.optimizer_step,
                "accumulation_step": self.state.accumulation_step,
                "gradient_accumulation_steps": grad_accum,
                "loss_total": None,
            }

        # An empty step contributes no bad loss, but it must still take part in
        # the agreement collective that train_step performs, or the two paths
        # issue different numbers of operations and NCCL desyncs.
        any_nonfinite = dist_utils.average_gradients(
            self.current_policy.parameters(), nonfinite_flag=False
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.current_policy.parameters(),
            self.config.optim.grad_clip_norm,
            error_if_nonfinite=False,
        )
        if any_nonfinite:
            self.state.skipped_steps += 1
            self.state.accumulation_step = 0
            self.optimizer.zero_grad(set_to_none=True)
            self._step_scheduler()
            return {"status": "skipped_nonfinite_loss_empty", "loss_total": None}
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._step_scheduler()
        self.state.optimizer_step += 1
        self.state.accumulation_step = 0
        self._maybe_update_old_policy()
        return {
            "status": "ok_empty",
            "optimizer_step": self.state.optimizer_step,
            "accumulation_step": 0,
            "gradient_accumulation_steps": grad_accum,
            "loss_total": None,
            "grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
            "lr": self._current_lr(),
        }

    def can_checkpoint(self) -> bool:
        return self.state.accumulation_step == 0

    def _batch_size(self, train_batch: Any) -> int:
        x0 = getattr(train_batch, "x0_onehot", None)
        if isinstance(x0, torch.Tensor) and x0.ndim >= 1:
            return int(x0.shape[0])
        return 1

    def _iter_microbatches(self, train_batch: Any, batch_size: int) -> list[tuple[Any, int]]:
        microbatch_size = max(1, int(self.config.optim.microbatch_size))
        if batch_size <= microbatch_size or not hasattr(train_batch, "select"):
            return [(train_batch, batch_size)]
        batches: list[tuple[Any, int]] = []
        for start in range(0, batch_size, microbatch_size):
            end = min(batch_size, start + microbatch_size)
            mask = torch.zeros(batch_size, dtype=torch.bool)
            mask[start:end] = True
            batches.append((train_batch.select(mask), end - start))
        return batches

    def _maybe_update_old_policy(self) -> None:
        interval = max(1, int(self.config.ema.update_interval))
        if self.state.optimizer_step % interval != 0:
            return
        decay = min(
            (1.0 + float(self.state.optimizer_step)) / (10.0 + float(self.state.optimizer_step)),
            float(self.config.ema.decay_max),
        )
        with torch.no_grad():
            for old_param, current_param in zip(self.old_policy.parameters(), self.current_policy.parameters()):
                old_param.data.mul_(decay).add_(current_param.data, alpha=1.0 - decay)
            for old_buffer, current_buffer in zip(self.old_policy.buffers(), self.current_policy.buffers()):
                old_buffer.copy_(current_buffer)
        self.state.ema_decay = decay

    def _compute_default_nft_loss(
        self,
        *,
        train_batch: Any,
        current_policy: torch.nn.Module,
        old_policy: torch.nn.Module,
        reference_policy: torch.nn.Module | None,
    ) -> Mapping[str, torch.Tensor]:
        from .losses import (
            DiffusionNFTLossConfig,
            compute_nft_epsilon_loss,
            forward_noise_x0,
            alpha_sigma_from_timesteps,
            sample_stratified_log_snr_timesteps,
        )
        from .model_adapter import RIDEModelAdapter

        batch = train_batch.to(self.device) if hasattr(train_batch, "to") else train_batch
        x0 = batch.x0_onehot
        if x0.ndim != 3:
            raise ValueError("TrainBatch.x0_onehot must have shape [B, L, 4]")
        batch_size, length, channels = x0.shape
        draws = int(self.config.loss.k_t)
        timesteps = sample_stratified_log_snr_timesteps(
            batch_size,
            draws,
            self.config.loss.log_snr_min,
            self.config.loss.log_snr_max,
            device=x0.device,
        )
        noise = torch.randn(batch_size, draws, length, channels, dtype=x0.dtype, device=x0.device)
        x0_draws = x0.unsqueeze(1).expand(-1, draws, -1, -1)
        z_t = forward_noise_x0(x0_draws, timesteps, noise)
        z_flat = z_t.reshape(batch_size * draws, length, channels)
        t_flat = timesteps.reshape(batch_size * draws)
        alpha_flat, sigma_flat = alpha_sigma_from_timesteps(t_flat)
        noise_level_flat = torch.log(alpha_flat.square() / sigma_flat.square())

        condition = batch.condition
        current_adapter = RIDEModelAdapter(current_policy)
        old_adapter = RIDEModelAdapter(old_policy)
        ref_adapter = RIDEModelAdapter(reference_policy) if reference_policy is not None else None

        with current_adapter.current_graph_cache(("trainer", self.state.optimizer_step)):
            current_encoding = current_adapter.encode_condition(condition, "current", str(self.state.optimizer_step))
            epsilon_current = current_adapter.predict_noise(current_encoding, z_flat, noise_level_flat, time=t_flat)
        with torch.no_grad():
            old_encoding = old_adapter.encode_condition(condition, "old", str(self.state.optimizer_step))
            epsilon_old = old_adapter.predict_noise(old_encoding, z_flat, noise_level_flat, time=t_flat)
            epsilon_reference = None
            if ref_adapter is not None:
                ref_encoding = ref_adapter.encode_condition(condition, "reference", "frozen")
                epsilon_reference = ref_adapter.predict_noise(ref_encoding, z_flat, noise_level_flat, time=t_flat)

        epsilon_current = epsilon_current.reshape(batch_size, draws, length, channels)
        epsilon_old = epsilon_old.reshape(batch_size, draws, length, channels)
        epsilon_reference = (
            epsilon_reference.reshape(batch_size, draws, length, channels)
            if epsilon_reference is not None
            else None
        )
        optimality = batch.optimality_probability.to(x0.device).unsqueeze(1).expand(-1, draws)
        valid_mask = batch.valid_mask.to(x0.device).unsqueeze(1).expand(-1, draws)
        cfg = DiffusionNFTLossConfig(
            nft_beta=self.config.loss.nft_beta,
            reference_weight=self.config.loss.reference_weight,
            timestep_weighting=self.config.loss.weighting,
            residual_normalization=self.config.loss.residual_normalization,
            beta_normalization=self.config.loss.divide_by_beta,
            x0_weight_max=self.config.loss.x0_weight_max,
        )
        result = compute_nft_epsilon_loss(
            epsilon_current=epsilon_current,
            epsilon_old=epsilon_old,
            epsilon_target=noise,
            optimality_probability=optimality,
            timesteps=timesteps,
            epsilon_reference=epsilon_reference,
            config=cfg,
            valid_mask=valid_mask,
        )
        return {
            "loss_total": result.total,
            "loss_policy": result.policy,
            "loss_positive": result.positive,
            "loss_negative": result.negative,
            "loss_reference": result.reference,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.__dict__.copy(),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key, value in state.get("state", {}).items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
