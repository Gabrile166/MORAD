"""Configuration loading for the standalone RIDE evaluation command."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "ride_ribodiffusion_eval.v1"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("evaluation config must be a YAML mapping")
    config = _expand_config_value(deepcopy(dict(raw)))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")

    sampling = _mapping(config, "sampling")
    n_samples = int(sampling.get("n_samples", 8))
    if n_samples < 1:
        raise ValueError("sampling.n_samples must be positive")
    if bool(sampling.get("paper_diversity", True)) and n_samples != 8:
        raise ValueError("paper_diversity=true requires sampling.n_samples=8")
    if int(sampling.get("target_limit", 1)) < 1:
        raise ValueError("sampling.target_limit must be positive")

    ride = _mapping(config, "ride")
    if not ride.get("rl_config"):
        raise ValueError("ride.rl_config is required")
    metrics = _mapping(config, "metrics")
    for name in ("sequence", "secondary", "tertiary", "rfam", "drfold"):
        _mapping(metrics, name)
    return config


def _expand_config_value(value: Any) -> Any:
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
        raise ValueError(f"environment variable {variable!r} is required by the evaluation config")

    return os.path.expanduser(_ENV_PATTERN.sub(replace, value))


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return dict(value)
