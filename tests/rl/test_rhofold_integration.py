from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from src.rl.protocol import TargetRecord
from src.rl.reward import JsonDiskCache, RewardComposer, RhoFoldJsonlClient


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = REPO_ROOT / "scripts" / "rhofold_oracle_worker.py"
RHOFOLD_PYTHON = Path(os.environ.get("RHOFOLD_PYTHON", "rhofold-python-not-configured"))
RHOFOLD_CKPT = REPO_ROOT.parent / "third_party" / "rhofold_protocol" / "checkpoints" / "rhofold_pretrained_params.pt"


def test_jsonl_worker_fake_backend_returns_artifact(tmp_path) -> None:
    client = RhoFoldJsonlClient([sys.executable, str(WORKER), "--fake", "--device", "cpu"], timeout_s=10.0)
    try:
        result = client.fold("AUGC", tmp_path / "out", target_id="target-1")
    finally:
        client.close()

    assert result.status == "ok"
    assert result.predicted_structure_path is not None
    assert Path(result.predicted_structure_path).exists()
    assert result.predicted_structure_hash is not None
    assert result.plddt == pytest.approx(0.77)


def test_jsonl_worker_fake_backend_structures_invalid_sequence(tmp_path) -> None:
    client = RhoFoldJsonlClient([sys.executable, str(WORKER), "--fake", "--device", "cpu"], timeout_s=10.0)
    try:
        result = client.fold("AUXC", tmp_path / "out", target_id="target-1")
    finally:
        client.close()

    assert result.status == "invalid_sequence"
    assert result.predicted_structure_path is None
    assert result.error_message


def test_jsonl_client_score_sequence_uses_fake_worker_and_metric_cache(tmp_path) -> None:
    target_pdb = tmp_path / "target.pdb"
    _write_linear_c4p_pdb(target_pdb, length=4)
    target = TargetRecord(
        target_id="target-1",
        sequence_native="AUGC",
        length=4,
        structure_ref=str(target_pdb),
        structure_hash="target-hash",
        split="train",
        source="unit",
        dataset_version="unit",
        native_oracle_metrics={},
        frozen_ride_metrics={},
        calibration_status="ok",
    )
    client = RhoFoldJsonlClient([sys.executable, str(WORKER), "--fake", "--device", "cpu"], timeout_s=10.0, output_root=tmp_path / "folds")
    metric_cache = JsonDiskCache(tmp_path / "metric-cache")
    try:
        first = client.score_sequence("AUGC", target, RewardComposer(), "oracle-key", metric_cache)
        second = client.score_sequence("AUGC", target, RewardComposer(), "oracle-key-2", metric_cache)
    finally:
        client.close()

    assert first.status == "ok"
    assert first.rmsd == pytest.approx(0.0)
    assert first.tm_score == pytest.approx(0.0)
    assert first.gdt_ts == pytest.approx(1.0)
    assert second.status == "ok"
    assert second.oracle_cache_hit


@pytest.mark.skipif(os.environ.get("RUN_RHOFOLD_REAL") != "1", reason="set RUN_RHOFOLD_REAL=1 to run real RhoFold+ integration")
def test_real_rhofold_plus_worker_single_sequence_smoke(tmp_path) -> None:
    if not RHOFOLD_PYTHON.exists():
        pytest.skip(f"missing RhoFold+ Python: {RHOFOLD_PYTHON}")
    if not RHOFOLD_CKPT.exists():
        pytest.skip(f"missing RhoFold+ checkpoint: {RHOFOLD_CKPT}")

    client = RhoFoldJsonlClient(
        [
            str(RHOFOLD_PYTHON),
            str(WORKER),
            "--ckpt",
            str(RHOFOLD_CKPT),
            "--device",
            "cuda:0",
            "--relax-steps",
            "0",
        ],
        timeout_s=120.0,
    )
    try:
        result = client.fold("AUGCAUGCAUGCAUGCAUGC", tmp_path / "real", target_id="real-smoke")
    finally:
        client.close()

    assert result.status == "ok"
    assert result.predicted_structure_path is not None
    assert Path(result.predicted_structure_path).exists()
    assert result.predicted_structure_hash


def _write_linear_c4p_pdb(path: Path, length: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(1, length + 1):
            handle.write(
                f"ATOM  {idx:5d}  C4'   A A{idx:4d}    {idx:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 77.00           C\n"
            )
        handle.write("END\n")
