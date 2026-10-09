"""Shared construction helpers for RIDE training and standalone evaluation."""

from __future__ import annotations

import sys
from typing import Any

import torch

from .config import RLConfig


def load_ride_model(config: RLConfig, device: torch.device) -> torch.nn.Module:
    import ml_collections

    from src.model import GVPDiff

    model_cfg = ml_collections.ConfigDict(config.ride_model.__dict__)
    model = GVPDiff(model_cfg).to(device)
    path = config.data.ride_checkpoint
    if not path:
        raise ValueError("data.ride_checkpoint is required")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Loaded RIDE checkpoint with missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    model.eval()
    return model


def build_noise_scheduler(config: RLConfig) -> Any:
    from src.noise_schedule import NoiseScheduleVP

    return NoiseScheduleVP(
        schedule=config.noise_schedule.schedule,
        continuous_beta_0=config.noise_schedule.continuous_beta_0,
        continuous_beta_1=config.noise_schedule.continuous_beta_1,
        eps=config.noise_schedule.eps,
    )
