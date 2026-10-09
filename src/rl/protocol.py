"""Typed data protocol shared by rollout, reward, and trainer modules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch


TensorMap = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class TargetRecord:
    target_id: str
    sequence_native: str
    length: int
    structure_ref: str
    structure_hash: str
    split: str
    source: str
    dataset_version: str
    native_oracle_metrics: Mapping[str, float]
    frozen_ride_metrics: Mapping[str, float]
    calibration_status: str
    reference_backbone_coords: torch.Tensor | None = None
    reference_c4p_coords: torch.Tensor | None = None
    reference_c1p_coords: torch.Tensor | None = None
    raw_record: Mapping[str, Any] | None = None
    dataset_index: int | None = None
    conformer_index: int | None = None

    def validate(self) -> "TargetRecord":
        if self.length != len(self.sequence_native):
            raise ValueError("target length must match sequence_native length")
        if self.length <= 0:
            raise ValueError("target length must be positive")
        if not self.target_id or not self.structure_hash:
            raise ValueError("target_id and structure_hash are required")
        if self.reference_backbone_coords is not None:
            if self.reference_backbone_coords.shape != (self.length, 3, 3):
                raise ValueError("reference_backbone_coords must have shape [L, 3, 3]")
            if not torch.isfinite(self.reference_backbone_coords).all():
                raise ValueError("reference_backbone_coords must be finite")
        if self.reference_c4p_coords is not None:
            if self.reference_c4p_coords.shape != (self.length, 3):
                raise ValueError("reference_c4p_coords must have shape [L, 3]")
            if not torch.isfinite(self.reference_c4p_coords).all():
                raise ValueError("reference_c4p_coords must be finite")
        if self.reference_c1p_coords is not None:
            if self.reference_c1p_coords.shape != (self.length, 3):
                raise ValueError("reference_c1p_coords must have shape [L, 3]")
            if not torch.isfinite(self.reference_c1p_coords).all():
                raise ValueError("reference_c1p_coords must be finite")
        return self


@dataclass(frozen=True)
class ConditionBundle:
    condition_id: str
    target_id: str
    round_id: int
    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    node_mask: torch.Tensor
    condition_noise_scale: float
    feature_hash: str
    pyg_data: Any = None

    def validate(self) -> "ConditionBundle":
        if self.condition_noise_scale != 0.0:
            raise ValueError("POC ConditionBundle requires condition_noise_scale=0.0")
        if self.node_features.ndim < 2:
            raise ValueError("node_features must have at least 2 dimensions")
        length = self.node_features.shape[-2]
        _validate_mask(self.node_mask, length, "node_mask")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.edge_features.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_features first dimension must match edge_index E")
        if not self.condition_id or not self.feature_hash:
            raise ValueError("condition_id and feature_hash are required")
        return self

    def to(self, device: torch.device | str) -> "ConditionBundle":
        return replace(
            self,
            node_features=self.node_features.to(device),
            edge_features=self.edge_features.to(device),
            edge_index=self.edge_index.to(device),
            node_mask=self.node_mask.to(device),
            pyg_data=self.pyg_data.to(device) if hasattr(self.pyg_data, "to") else self.pyg_data,
        )

    def cpu(self) -> "ConditionBundle":
        return self.to("cpu")


@dataclass(frozen=True)
class RolloutSample:
    sample_id: str
    target_id: str
    round_id: int
    tokens: str
    x0_onehot: torch.Tensor
    seed: int
    temperature: float
    policy_version: str
    condition_id: str
    sampler_name: str
    sampler_version: str
    latency_ms: float

    def validate(self) -> "RolloutSample":
        _validate_onehot(self.x0_onehot, "x0_onehot")
        if len(self.tokens) != self.x0_onehot.shape[0]:
            raise ValueError("tokens length must match x0_onehot length")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        return self


@dataclass(frozen=True)
class RewardRecord:
    status: str
    rmsd: float | None
    tm_score: float | None
    gdt_ts: float | None
    raw_reward: float | None
    is_good: bool
    oracle_cache_hit: bool
    oracle_latency_ms: float
    oracle_key: str
    error_message: str | None = None
    plddt: float | None = None
    # geometric_v3 auxiliary terms; None when the preset does not compute them.
    # Not required at status=ok -- composite() renormalises over what exists.
    sc_mcc: float | None = None
    ensemble_defect: float | None = None
    composition_kl: float | None = None

    def validate(self) -> "RewardRecord":
        if self.status == "ok":
            for name in ("rmsd", "tm_score", "gdt_ts", "raw_reward"):
                if getattr(self, name) is None:
                    raise ValueError(f"{name} is required when status=ok")
        if self.oracle_latency_ms < 0:
            raise ValueError("oracle_latency_ms must be non-negative")
        return self


@dataclass(frozen=True)
class RolloutBatch:
    samples: tuple[RolloutSample, ...]
    condition: ConditionBundle

    def validate(self) -> "RolloutBatch":
        self.condition.validate()
        for sample in self.samples:
            sample.validate()
            if sample.condition_id != self.condition.condition_id:
                raise ValueError("sample condition_id does not match batch condition")
            if sample.target_id != self.condition.target_id:
                raise ValueError("sample target_id does not match batch condition")
        return self

    def select(self, mask: torch.Tensor | Sequence[bool]) -> "RolloutBatch":
        mask_list = _mask_to_list(mask, len(self.samples))
        return replace(self, samples=tuple(item for item, keep in zip(self.samples, mask_list) if keep))

    def to(self, device: torch.device | str) -> "RolloutBatch":
        samples = tuple(replace(sample, x0_onehot=sample.x0_onehot.to(device)) for sample in self.samples)
        return replace(self, samples=samples, condition=self.condition.to(device))

    def cpu(self) -> "RolloutBatch":
        return self.to("cpu")


@dataclass(frozen=True)
class RewardBatch:
    records: tuple[RewardRecord, ...]
    target_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]

    def validate(self) -> "RewardBatch":
        if len(self.records) != len(self.target_ids) or len(self.records) != len(self.sample_ids):
            raise ValueError("records, target_ids, and sample_ids lengths must match")
        for record in self.records:
            record.validate()
        return self

    def select(self, mask: torch.Tensor | Sequence[bool]) -> "RewardBatch":
        mask_list = _mask_to_list(mask, len(self.records))
        return RewardBatch(
            records=tuple(item for item, keep in zip(self.records, mask_list) if keep),
            target_ids=tuple(item for item, keep in zip(self.target_ids, mask_list) if keep),
            sample_ids=tuple(item for item, keep in zip(self.sample_ids, mask_list) if keep),
        )

    def to(self, device: torch.device | str) -> "RewardBatch":
        return self

    def cpu(self) -> "RewardBatch":
        return self


@dataclass(frozen=True)
class TrainBatch:
    x0_onehot: torch.Tensor
    condition: ConditionBundle
    optimality_probability: torch.Tensor
    raw_reward: torch.Tensor
    advantage: torch.Tensor
    target_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    policy_versions: tuple[str, ...]
    condition_ids: tuple[str, ...]
    valid_mask: torch.Tensor
    skip_update: bool = False
    skip_reason: str | None = None

    def validate(self) -> "TrainBatch":
        if self.x0_onehot.ndim != 3 or self.x0_onehot.shape[-1] != 4:
            raise ValueError("x0_onehot must have shape [B, L, 4]")
        _validate_onehot(self.x0_onehot, "x0_onehot")
        batch = self.x0_onehot.shape[0]
        for name, value in (
            ("optimality_probability", self.optimality_probability),
            ("raw_reward", self.raw_reward),
            ("advantage", self.advantage),
            ("valid_mask", self.valid_mask),
        ):
            if value.shape != (batch,):
                raise ValueError(f"{name} must have shape [B]")
        if not torch.all((0 <= self.optimality_probability) & (self.optimality_probability <= 1)):
            raise ValueError("optimality_probability must be in [0, 1]")
        _validate_mask(self.valid_mask, batch, "valid_mask")
        for name, seq in (
            ("target_ids", self.target_ids),
            ("sample_ids", self.sample_ids),
            ("policy_versions", self.policy_versions),
            ("condition_ids", self.condition_ids),
        ):
            if len(seq) != batch:
                raise ValueError(f"{name} length must match batch size")
        if any(condition_id != self.condition.condition_id for condition_id in self.condition_ids):
            raise ValueError("all condition_ids must match condition.condition_id")
        if any(target_id != self.condition.target_id for target_id in self.target_ids):
            raise ValueError("all target_ids must match condition.target_id")
        self.condition.validate()
        return self

    def select(self, mask: torch.Tensor | Sequence[bool]) -> "TrainBatch":
        mask_tensor = _mask_to_tensor(mask, self.x0_onehot.shape[0], self.x0_onehot.device)
        mask_list = mask_tensor.cpu().tolist()
        return replace(
            self,
            x0_onehot=self.x0_onehot[mask_tensor],
            optimality_probability=self.optimality_probability[mask_tensor],
            raw_reward=self.raw_reward[mask_tensor],
            advantage=self.advantage[mask_tensor],
            valid_mask=self.valid_mask[mask_tensor],
            target_ids=tuple(item for item, keep in zip(self.target_ids, mask_list) if keep),
            sample_ids=tuple(item for item, keep in zip(self.sample_ids, mask_list) if keep),
            policy_versions=tuple(item for item, keep in zip(self.policy_versions, mask_list) if keep),
            condition_ids=tuple(item for item, keep in zip(self.condition_ids, mask_list) if keep),
        )

    def to(self, device: torch.device | str) -> "TrainBatch":
        return replace(
            self,
            x0_onehot=self.x0_onehot.to(device),
            condition=self.condition.to(device),
            optimality_probability=self.optimality_probability.to(device),
            raw_reward=self.raw_reward.to(device),
            advantage=self.advantage.to(device),
            valid_mask=self.valid_mask.to(device),
        )

    def cpu(self) -> "TrainBatch":
        return self.to("cpu")


def train_batch_from_rollout(
    rollout: RolloutBatch,
    rewards: RewardBatch,
    optimality_probability: torch.Tensor,
    advantage: torch.Tensor,
) -> TrainBatch:
    rollout.validate()
    rewards.validate()
    if len(rollout.samples) != len(rewards.records):
        raise ValueError("rollout and reward sizes must match")
    return TrainBatch(
        x0_onehot=torch.stack([sample.x0_onehot for sample in rollout.samples], dim=0),
        condition=rollout.condition,
        optimality_probability=optimality_probability,
        raw_reward=torch.tensor(
            [float(record.raw_reward) if record.raw_reward is not None else float("nan") for record in rewards.records],
            dtype=torch.float32,
            device=optimality_probability.device,
        ),
        advantage=advantage,
        target_ids=tuple(sample.target_id for sample in rollout.samples),
        sample_ids=tuple(sample.sample_id for sample in rollout.samples),
        policy_versions=tuple(sample.policy_version for sample in rollout.samples),
        condition_ids=tuple(sample.condition_id for sample in rollout.samples),
        valid_mask=torch.tensor([record.status == "ok" for record in rewards.records], dtype=torch.bool),
    ).validate()


def _validate_onehot(value: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a floating point tensor")
    if value.shape[-1] != 4:
        raise ValueError(f"{name} last dimension must be 4")
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} values must be exactly 0 or 1")
    if not torch.all(value.sum(dim=-1) == 1):
        raise ValueError(f"{name} rows must sum to 1")


def _validate_mask(value: torch.Tensor, length: int, name: str) -> None:
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be bool")
    if value.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")


def _mask_to_tensor(mask: torch.Tensor | Sequence[bool], length: int, device: torch.device | str) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        tensor = mask.to(device=device)
    else:
        tensor = torch.tensor(list(mask), dtype=torch.bool, device=device)
    _validate_mask(tensor, length, "selection mask")
    return tensor


def _mask_to_list(mask: torch.Tensor | Sequence[bool], length: int) -> list[bool]:
    return _mask_to_tensor(mask, length, "cpu").cpu().tolist()
