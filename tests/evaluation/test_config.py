from __future__ import annotations

import pytest
import yaml

from src.evaluation.config import SCHEMA_VERSION, load_evaluation_config


def test_load_evaluation_config_accepts_paper_diversity_with_eight_samples(tmp_path) -> None:
    path = tmp_path / "evaluation.yaml"
    _write_config(path, n_samples=8, paper_diversity=True)

    config = load_evaluation_config(path)

    assert config["sampling"]["n_samples"] == 8
    assert config["sampling"]["paper_diversity"] is True


def test_load_evaluation_config_rejects_paper_diversity_without_eight_samples(tmp_path) -> None:
    path = tmp_path / "evaluation.yaml"
    _write_config(path, n_samples=4, paper_diversity=True)

    with pytest.raises(ValueError, match="paper_diversity=true requires sampling.n_samples=8"):
        load_evaluation_config(path)


def test_load_evaluation_config_allows_non_paper_diversity_sample_count(tmp_path) -> None:
    path = tmp_path / "evaluation.yaml"
    _write_config(path, n_samples=4, paper_diversity=False)

    config = load_evaluation_config(path)

    assert config["sampling"]["n_samples"] == 4
    assert config["sampling"]["paper_diversity"] is False


def test_load_evaluation_config_expands_environment_values_and_defaults(tmp_path, monkeypatch) -> None:
    path = tmp_path / "evaluation.yaml"
    _write_config(path, n_samples=8, paper_diversity=True)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["ride"]["pool_manifest"] = "${TEST_POOL_PATH}"
    payload["metrics"]["tertiary"]["rhofold_python"] = "${MISSING_RHOFOLD_PYTHON:-/fallback/python}"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setenv("TEST_POOL_PATH", "/tmp/pool.pt")

    config = load_evaluation_config(path)

    assert config["ride"]["pool_manifest"] == "/tmp/pool.pt"
    assert config["metrics"]["tertiary"]["rhofold_python"] == "/fallback/python"


def test_load_evaluation_config_rejects_missing_required_environment_value(tmp_path) -> None:
    path = tmp_path / "evaluation.yaml"
    _write_config(path, n_samples=8, paper_diversity=True)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["ride"]["pool_manifest"] = "${RIDE_EVAL_REQUIRED_TEST_VALUE}"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="RIDE_EVAL_REQUIRED_TEST_VALUE"):
        load_evaluation_config(path)


def _write_config(path, *, n_samples: int, paper_diversity: bool) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sampling": {
            "n_samples": n_samples,
            "paper_diversity": paper_diversity,
            "target_limit": 1,
        },
        "ride": {"rl_config": "configs/rl.yaml"},
        "metrics": {
            "sequence": {},
            "secondary": {},
            "tertiary": {},
            "rfam": {},
            "drfold": {},
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
