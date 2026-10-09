from pathlib import Path

import pytest

from src.rl.config import ConfigError, ResidualNormalization, TimestepWeighting, load_config, parse_config


def test_parse_nested_config_and_enums():
    config = parse_config(
        {
            "data": {"pool_manifest": "data/rl_pool/demo/targets.jsonl"},
            "rollout": {"group_size": 8},
            "trainer": {
                "loss": {
                    "nft_beta": 0.2,
                    "k_t": 4,
                    "weighting": "x0_reconstruction",
                    "residual_normalization": "mean_abs_detached",
                },
            },
        }
    )

    assert config.rollout.group_size == 8
    assert config.trainer.loss.k_t == 4
    assert config.trainer.loss.weighting is TimestepWeighting.X0_RECONSTRUCTION
    assert config.trainer.loss.residual_normalization is ResidualNormalization.MEAN_ABS_DETACHED


def test_unknown_nested_key_fails_fast():
    with pytest.raises(ConfigError, match="unknown config key"):
        parse_config({"trainer": {"loss": {"surprise": True}}})


def test_train_mode_rejects_compat_reward():
    with pytest.raises(ConfigError, match="current_code_compat"):
        parse_config({"reward": {"composer": {"preset": "current_code_compat"}}})


def test_condition_noise_requires_snapshot():
    with pytest.raises(ConfigError, match="condition_snapshot"):
        parse_config({"data": {"condition_noise_scale": 0.1}})


def test_loss_fail_fast_rules():
    with pytest.raises(ConfigError, match="nft_beta"):
        parse_config({"trainer": {"loss": {"nft_beta": 0.0}}})
    with pytest.raises(ConfigError, match="group_size"):
        parse_config({"rollout": {"group_size": 1}})
    with pytest.raises(ConfigError, match="x0_representation"):
        parse_config({"data": {"x0_representation": "latent"}})


def test_load_config_expands_environment_and_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RIDE_TEST_CHECKPOINT", "/models/ride.h5")
    config_path = tmp_path / "rl.yaml"
    config_path.write_text(
        """
data:
  ride_checkpoint: "${RIDE_TEST_CHECKPOINT}"
  pool_manifest: "${RIDE_TEST_POOL:-artifacts/default.pt}"
reward:
  oracle:
    python: "${RIDE_TEST_RHOFOLD_PYTHON:-python}"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.ride_checkpoint == "/models/ride.h5"
    assert config.data.pool_manifest == "artifacts/default.pt"
    assert config.reward.oracle.python == "python"


def test_load_config_requires_environment_variable_without_default(tmp_path: Path):
    config_path = tmp_path / "rl.yaml"
    config_path.write_text('data:\n  ride_checkpoint: "${RIDE_MISSING_FOR_TEST}"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="RIDE_MISSING_FOR_TEST"):
        load_config(config_path)
