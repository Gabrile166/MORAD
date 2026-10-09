"""Small typed core for MORAD training."""

from .advantage import AdvantageResult, EmaRewardScale, WindowRewardScale, compute_group_advantages
from .config import RLConfig, load_config, parse_config
from .losses import (
    DiffusionNFTLossConfig,
    DiffusionNFTLossResult,
    compute_nft_epsilon_loss,
    sample_stratified_log_snr_timesteps,
    timestep_weights,
)
from .protocol import (
    ConditionBundle,
    RewardBatch,
    RewardRecord,
    RolloutBatch,
    RolloutSample,
    TargetRecord,
    TrainBatch,
)

__all__ = [
    "AdvantageResult",
    "ConditionBundle",
    "DiffusionNFTLossConfig",
    "DiffusionNFTLossResult",
    "EmaRewardScale",
    "RLConfig",
    "RewardBatch",
    "RewardRecord",
    "RolloutBatch",
    "RolloutSample",
    "TargetRecord",
    "TrainBatch",
    "WindowRewardScale",
    "compute_group_advantages",
    "compute_nft_epsilon_loss",
    "load_config",
    "parse_config",
    "sample_stratified_log_snr_timesteps",
    "timestep_weights",
]
