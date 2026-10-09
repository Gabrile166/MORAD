from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from src.rl.targets import TargetPoolConfig, build_target_pool, write_manifest


class GoodCalibration:
    name = "fake-rhofold-calibration"

    def evaluate(self, target):
        return {"status": "ok", "backend": self.name, "rmsd": 1.0, "tm_score": 0.5, "gdt": 0.6}


class GoodBaseline:
    name = "fake-frozen-ride"

    def evaluate(self, target):
        return {"status": "ok", "backend": self.name, "rmsd": 2.0, "tm_score": 0.45, "gdt": 0.5}


def test_cli_train_reaches_one_optimizer_update_on_production_modules(tmp_path: Path) -> None:
    manifest_path = tmp_path / "targets.pt"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "run"
    manifest = build_target_pool(
        {"demo": _raw_record("ACGU" * 5)},
        {"train": [0], "validation": [], "test": []},
        TargetPoolConfig(limit=1, dev_mode=True),
        GoodCalibration(),
        GoodBaseline(),
    )
    write_manifest(manifest, manifest_path)
    config = {
        "data": {
            "pool_manifest": str(manifest_path),
            "condition_noise_scale": 0.0,
            "condition_snapshot_enabled": False,
            "x0_representation": "one_hot_01",
        },
        "runtime": {
            "seed": 123,
            "device": "cpu",
            "mode": "train",
            "dev_tiny_policy": True,
            "budget": {
                "max_outer_steps": 1,
                "max_oracle_calls": 8,
                "max_wall_hours": 0.1,
                "checkpoint_every_oracle_calls": 8,
            },
        },
        "rollout": {"group_size": 2, "sampler": {"n_steps": 4}, "temperature": {"initial": 0.2}},
        "reward": {
            "oracle": {
                "backend": "local_jsonl_worker",
                "python": sys.executable,
                "worker_script": "scripts/rhofold_oracle_worker.py",
                "output_dir": str(tmp_path / "folds"),
                "timeout_sec": 10,
                "device": "cpu",
                "fake": True,
            },
            "metric_cache": {"dir": str(tmp_path / "cache")},
            "composer": {"preset": "initial_plan_strict"},
        },
        "advantage": {"scale": {"kind": "ema", "decay": 0.0, "epsilon": 1.0e-6, "min_valid": 2}},
        "trainer": {
            "loss": {"nft_beta": 0.1, "reference_weight": 0.0, "k_t": 2, "weighting": "uniform_eps"},
            "optim": {"lr": 1.0e-2, "grad_clip_norm": 10.0, "amp": False, "microbatch_size": 1},
            "ema": {"decay_max": 0.995, "update_interval": 1},
        },
        "noise_schedule": {"schedule": "linear", "continuous_beta_0": 0.1, "continuous_beta_1": 0.2, "eps": 0.001},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "train.py", "--mode", "train", "--config", str(config_path), "--output-dir", str(output_dir)],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    summary = json.loads(proc.stdout.strip().splitlines()[-1])

    assert summary["outer_step"] == 1
    assert summary["oracle_calls"] >= 1
    checkpoint = torch.load(output_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
    assert checkpoint["cursor"]["optimizer_step"] == 1


def _raw_record(sequence: str) -> dict[str, object]:
    return {"sequence": sequence, "id_list": ["demo"], "coords_list": [_full_atom_coords(len(sequence))]}


def _full_atom_coords(length: int) -> torch.Tensor:
    coords = torch.ones(length, 27, 3, dtype=torch.float32) * 1.0e-5
    for idx in range(length):
        base = float(idx + 1)
        coords[idx, 0] = torch.tensor([base, 0.0, 0.0])
        coords[idx, 3] = torch.tensor([base, 1.0, 0.0])
        coords[idx, 10] = torch.tensor([base, 1.0, 1.0])
        coords[idx, 24] = torch.tensor([base, 1.0, -1.0])
    return coords
