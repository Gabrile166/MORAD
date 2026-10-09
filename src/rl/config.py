"""Strict nested configuration for the lightweight MORAD stack."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_args, get_origin, get_type_hints

import yaml

# Paper names for the reward presets. `morad` is the six-component
# weighted-geometric-mean reward; `structural_only` is the GDT-TS/TM-score/RMSD
# control reward of the reward comparison. The internal names stay valid.
PRESET_ALIASES = {"morad": "geometric_v3", "structural_only": "balanced_cosine_v1"}


def canonical_preset(name: str) -> str:
    return PRESET_ALIASES.get(name, name)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    """Raised when a config is incomplete, unknown, or unsafe."""


class TimestepWeighting(str, Enum):
    UNIFORM_EPS = "uniform_eps"
    VELOCITY_EQUIVALENT = "velocity_equivalent"
    X0_RECONSTRUCTION = "x0_reconstruction"


class ResidualNormalization(str, Enum):
    NONE = "none"
    MEAN_ABS_DETACHED = "mean_abs_detached"


class AdvantageScaleKind(str, Enum):
    EMA = "ema"
    WINDOW = "window"


@dataclass(frozen=True)
class DataConfig:
    pool_manifest: str | None = None
    ride_checkpoint: str | None = None
    processed_data: str = "../data/processed.pt"
    split_file: str = "../data/das_split.pt"
    condition_noise_scale: float = 0.0
    condition_snapshot_enabled: bool = False
    x0_representation: str = "one_hot_01"


@dataclass(frozen=True)
class SamplerConfig:
    name: str = "x0_renoise"
    n_steps: int = 50
    version: str = "v1"


@dataclass(frozen=True)
class TemperatureConfig:
    initial: float = 1.0
    min: float = 0.5
    max: float = 1.2
    step: float = 0.1


@dataclass(frozen=True)
class RescueConfig:
    max_rounds: int = 0
    min_unique_sequences: int = 2
    min_valid_sequences: int = 2


@dataclass(frozen=True)
class RolloutConfig:
    group_size: int = 2
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    rescue: RescueConfig = field(default_factory=RescueConfig)


@dataclass(frozen=True)
class OracleConfig:
    backend: str = "local_jsonl_worker"
    python: str = "python"
    worker_script: str = "scripts/rhofold_oracle_worker.py"
    ckpt: str | None = None
    output_dir: str = "outputs/rhofold_predictions"
    timeout_sec: float = 300.0
    device: str = "cuda:0"
    fake: bool = False
    checkpoint_hash: str | None = None


@dataclass(frozen=True)
class MetricCacheConfig:
    enabled: bool = True
    dir: str = "cache/rl_metrics"
    metric_version: str = "ride_evaluator_v1"
    alignment_config: str = "c4p_kabsch_centered"


@dataclass(frozen=True)
class RewardComposerConfig:
    preset: str = "initial_plan_strict"
    good: dict[str, float] = field(default_factory=lambda: {"gdt_ts": 0.5, "tm_score": 0.45, "rmsd": 2.0})


@dataclass(frozen=True)
class RewardConfig:
    oracle: OracleConfig = field(default_factory=OracleConfig)
    metric_cache: MetricCacheConfig = field(default_factory=MetricCacheConfig)
    composer: RewardComposerConfig = field(default_factory=RewardComposerConfig)


@dataclass(frozen=True)
class AdvantageScaleConfig:
    kind: AdvantageScaleKind = AdvantageScaleKind.EMA
    epsilon: float = 1.0e-6
    decay: float = 0.95
    window_size: int = 128
    min_valid: int = 2


@dataclass(frozen=True)
class ValidationConfig:
    """In-loop evaluation on a held-out set."""

    enabled: bool = False
    config: str = ""
    target_limit: int = 0
    # Counted in optimizer updates, matching the step definition.
    test_freq: int = 0
    before_train: bool = False
    # Which pool to validate on. Kept separate from `data.pool_manifest` so the
    # held-out set cannot be confused with the training pool.
    pool_manifest: str = ""
    # Comma-separated devices the validation shards fold on. Folding is the
    # bottleneck and is what gets parallelised; leave empty to fold on the
    # training device, which is serial and only sensible for small pools.
    devices: str = ""
    n_samples: int = 8
    timeout_sec: float = 5400.0
    usalign_binary: str = "../third_party/USalign/USalign"
    # Policy snapshots kept on disk beyond best, runner-up and latest
    # (0 keeps every snapshot).
    keep_runs: int = 3


@dataclass(frozen=True)
class AdvantageConfig:
    scale: AdvantageScaleConfig = field(default_factory=AdvantageScaleConfig)
    # Standard GRPO: z-score within the rollout group, clipped.
    group_normalize: bool = True
    z_clip: float = 2.0


@dataclass(frozen=True)
class LossConfig:
    nft_beta: float = 0.1
    reference_weight: float = 0.0
    k_t: int = 8
    weighting: TimestepWeighting = TimestepWeighting.UNIFORM_EPS
    residual_normalization: ResidualNormalization = ResidualNormalization.NONE
    divide_by_beta: bool = True
    x0_weight_max: float = 100.0
    log_snr_min: float = -10.0
    log_snr_max: float = 10.0


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 1.0e-5
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    amp: bool = False
    microbatch_size: int = 1
    # LR schedule. "constant" keeps the old behaviour; "warmup_cosine" ramps up
    # over `lr_warmup_ratio` of training then decays to `lr * lr_min_ratio`.
    lr_schedule: str = "constant"
    lr_warmup_ratio: float = 0.05
    lr_min_ratio: float = 0.1
    lr_total_steps: int = 0


@dataclass(frozen=True)
class PolicyEmaConfig:
    decay_max: float = 0.999
    update_interval: int = 1


@dataclass(frozen=True)
class TrainerConfig:
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    ema: PolicyEmaConfig = field(default_factory=PolicyEmaConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 0
    device: str = "cuda"
    mode: str = "train"
    dev_tiny_policy: bool = False
    # How many ranks cooperate on one rollout target. world_size must be a
    # multiple of this; batch size in targets is world_size / ranks_per_target.
    ranks_per_target: int = 1
    # Draw targets in shuffled order instead of the curriculum's own ordering.
    shuffle_targets: bool = False
    shuffle_seed: int = 0
    # Passes over the target pool. The engine walks its target list exactly once,
    # so extra epochs are realised by repeating the pool (reshuffled per pass)
    # rather than by raising max_outer_steps, which the walk length would clamp.
    epochs: int = 1
    budget: dict[str, int | float] = field(default_factory=lambda: {
        "max_outer_steps": 1,
        "max_oracle_calls": 100,
        "max_wall_hours": 1.0,
        "checkpoint_every_oracle_calls": 25,
    })


@dataclass(frozen=True)
class LoggingConfig:
    jsonl: str = "runs/rl/events.jsonl"
    wandb: dict[str, Any] = field(default_factory=lambda: {"enabled": False})


@dataclass(frozen=True)
class CheckpointConfig:
    dir: str = "runs/rl/checkpoints"
    resume: str | None = None
    strict_hash: bool = True
    save_latest: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    deterministic_anchor_temperature: float = 0.0
    group_size: int = 2


@dataclass(frozen=True)
class NoiseScheduleConfig:
    schedule: str = "linear"
    continuous_beta_0: float = 0.1
    continuous_beta_1: float = 20.0
    eps: float = 1.0e-3


@dataclass(frozen=True)
class RideModelConfig:
    node_in_dim: tuple[int, int] = (15, 4)
    node_h_dim: tuple[int, int] = (256, 24)
    edge_in_dim: tuple[int, int] = (131, 3)
    edge_h_dim: tuple[int, int] = (128, 4)
    num_layers: int = 5
    drop_rate: float = 0.5
    out_dim: int = 4
    time_cond: bool = True


@dataclass(frozen=True)
class RLConfig:
    data: DataConfig = field(default_factory=DataConfig)
    ride_model: RideModelConfig = field(default_factory=RideModelConfig)
    noise_schedule: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def validate(self) -> "RLConfig":
        if self.trainer.loss.nft_beta <= 0:
            raise ConfigError("trainer.loss.nft_beta must be > 0")
        if self.rollout.group_size < 2:
            raise ConfigError("rollout.group_size must be >= 2")
        preset = canonical_preset(self.reward.composer.preset)
        if preset not in {"initial_plan_strict", "current_code_compat", "balanced_cosine_v1", "geometric_v3"}:
            raise ConfigError(
                "reward.composer.preset must be one of: morad, structural_only, initial_plan_strict, "
                "current_code_compat, balanced_cosine_v1, geometric_v3"
            )
        if preset == "current_code_compat" and self.runtime.mode == "train":
            raise ConfigError("current_code_compat reward preset is forbidden in train mode")
        if self.data.condition_noise_scale != 0 and not self.data.condition_snapshot_enabled:
            raise ConfigError("nonzero condition_noise_scale requires condition_snapshot_enabled=true")
        if self.data.x0_representation != "one_hot_01":
            raise ConfigError("data.x0_representation must be one_hot_01")
        if self.trainer.loss.k_t < 1:
            raise ConfigError("trainer.loss.k_t must be >= 1")
        if self.trainer.optim.microbatch_size < 1:
            raise ConfigError("trainer.optim.microbatch_size must be >= 1")
        if self.trainer.loss.log_snr_min >= self.trainer.loss.log_snr_max:
            raise ConfigError("trainer.loss.log_snr_min must be < log_snr_max")
        if self.trainer.loss.x0_weight_max <= 0:
            raise ConfigError("trainer.loss.x0_weight_max must be > 0")
        if not 0 <= self.advantage.scale.decay < 1:
            raise ConfigError("advantage.scale.decay must be in [0, 1)")
        if self.advantage.scale.window_size < 2:
            raise ConfigError("advantage.scale.window_size must be >= 2")
        if self.advantage.scale.epsilon <= 0:
            raise ConfigError("advantage.scale.epsilon must be > 0")
        if self.rollout.rescue.max_rounds > 2:
            raise ConfigError("rollout.rescue.max_rounds must be <= 2")
        if self.trainer.optim.gradient_accumulation_steps < 1:
            raise ConfigError("trainer.optim.gradient_accumulation_steps must be >= 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


T = TypeVar("T")


def load_config(path: str | Path) -> RLConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError("root config must be a mapping")
    return parse_config(_expand_config_value(raw))


def parse_config(raw: Mapping[str, Any]) -> RLConfig:
    return _parse_dataclass(RLConfig, raw, "root").validate()


def _expand_config_value(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in nested YAML values."""
    if isinstance(value, Mapping):
        return {key: _expand_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_value(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        variable, default = match.group(1), match.group(2)
        if variable in os.environ:
            return os.environ[variable]
        if default is not None:
            return default
        raise ConfigError(f"environment variable {variable!r} is required by the RL config")

    return os.path.expanduser(_ENV_PATTERN.sub(replace, value))


def _parse_dataclass(cls: type[T], raw: Mapping[str, Any], path: str) -> T:
    known = {item.name: item for item in fields(cls)}
    type_hints = get_type_hints(cls)
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"unknown config key(s) at {path}: {', '.join(unknown)}")

    values: dict[str, Any] = {}
    for name, item in known.items():
        if name not in raw:
            continue
        values[name] = _coerce(type_hints[name], raw[name], f"{path}.{name}")
    return cls(**values)


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    if origin is None and isinstance(annotation, str):
        return value
    if origin is not None and type(None) in get_args(annotation):
        if value is None:
            return None
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _coerce(non_none[0], value, path)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        return _parse_dataclass(annotation, value, path)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in annotation)
            raise ConfigError(f"{path} must be one of: {allowed}") from exc
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be a bool")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{path} must be an int")
        return value
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"{path} must be a float")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list/tuple")
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], item, f"{path}[]") for item in value)
        if len(value) != len(args):
            raise ConfigError(f"{path} must contain {len(args)} items")
        return tuple(_coerce(arg, item, f"{path}[{idx}]") for idx, (arg, item) in enumerate(zip(args, value)))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        key_type, val_type = get_args(annotation) or (str, Any)
        return {
            _coerce(key_type, key, f"{path}.<key>") if key_type is not Any else key:
            _coerce(val_type, item, f"{path}.{key}") if val_type is not Any else item
            for key, item in value.items()
        }
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value
