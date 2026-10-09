from __future__ import annotations

import torch

from src.rl.targets import (
    JsonlMetricsBackend,
    TargetPoolConfig,
    build_target_pool,
    load_target_pool,
    make_target,
    materialize_condition,
    stable_target_id,
    structure_digest,
    write_manifest,
)


class FakeCalibration:
    name = "fake"

    def __init__(self, status="ok"):
        self.status = status

    def evaluate(self, target):
        if self.status != "ok":
            return {"status": self.status, "backend": self.name}
        return {"status": "ok", "backend": self.name, "rmsd": 1.2, "tm_score": 0.55, "gdt": 0.6}


class FakeBaseline:
    name = "fake"

    def evaluate(self, target):
        return {"status": "ok", "backend": self.name, "sequence_recovery": 0.25}


def full_atom_coords(length: int) -> torch.Tensor:
    coords = torch.ones(length, 27, 3, dtype=torch.float32) * 1.0e-5
    for i in range(length):
        base = float(i)
        coords[i, 0] = torch.tensor([base, 0.0, 0.0])
        coords[i, 3] = torch.tensor([base, 1.0, 0.0])
        coords[i, 10] = torch.tensor([base, 1.0, 1.0])
        coords[i, 24] = torch.tensor([base, 1.0, -1.0])
    return coords


def raw_record(sequence: str, coords=None):
    return {
        "sequence": sequence,
        "id_list": ["1ABC_1_A"],
        "coords_list": [coords if coords is not None else full_atom_coords(len(sequence))],
    }


def test_make_target_is_stable_and_uses_backbone_coords():
    config = TargetPoolConfig()
    sequence = "ACGU" * 5
    target, reason = make_target(raw_record(sequence), 7, config)

    assert reason is None
    assert target is not None
    assert target.length == 20
    assert target.ref_backbone_coords.shape == (20, 3, 3)
    assert target.ref_c4p_coords.shape == (20, 3)
    assert target.mask_coords.all()
    assert target.structure_hash == structure_digest(sequence, target.ref_c4p_coords)
    assert target.target_id == stable_target_id("train", 7, sequence, target.structure_hash)


def test_build_target_pool_filters_test_leakage_and_requires_calibration():
    sequence = "ACGU" * 5
    processed = {sequence: raw_record(sequence)}
    split = {"train": [0], "validation": [], "test": [0]}
    config = TargetPoolConfig(require_calibrated=True, dev_mode=True)

    manifest = build_target_pool(processed, split, config, FakeCalibration(), FakeBaseline())

    assert manifest["targets"] == []
    assert manifest["summary"]["rejected"] == {"test_leakage": 1}


def test_build_target_pool_rejects_uncalibrated_and_keeps_original_multiplicity():
    good = "ACGU" * 5
    short = "ACGU"
    processed = {
        good: raw_record(good),
        short: raw_record(short),
    }
    split = {"train": [0, 1], "validation": [], "test": []}
    config = TargetPoolConfig(require_calibrated=True, dev_mode=True)

    manifest = build_target_pool(processed, split, config, FakeCalibration(status="uncalibrated"), FakeBaseline())

    assert manifest["targets"] == []
    assert manifest["summary"]["rejected"] == {"calibration_uncalibrated": 1, "too_short": 1}


def test_build_target_pool_serializes_expected_schema():
    good = "ACGU" * 5
    processed = {good: raw_record(good)}
    split = {"train": [0], "validation": [], "test": []}

    manifest = build_target_pool(
        processed,
        split,
        TargetPoolConfig(limit=1, dev_mode=True),
        FakeCalibration(),
        FakeBaseline(),
    )

    assert manifest["schema_version"] == "ride_rl_target_pool.v1"
    assert manifest["summary"]["selected"] == 1
    target = manifest["targets"][0]
    assert set(target) == {"metadata", "raw_record", "ref_backbone_coords", "ref_c4p_coords", "mask_coords", "calibration", "frozen_baseline"}
    assert target["metadata"]["dataset_index"] == 0
    assert target["calibration"]["status"] == "ok"
    assert target["frozen_baseline"]["status"] == "ok"


def test_production_pool_rejects_placeholder_backends():
    good = "ACGU" * 5
    processed = {good: raw_record(good)}
    split = {"train": [0], "validation": [], "test": []}

    try:
        build_target_pool(processed, split, TargetPoolConfig(limit=1), FakeCalibration(), FakeBaseline())
    except ValueError as exc:
        assert "dev placeholders" in str(exc)
    else:
        raise AssertionError("placeholder production pool should fail")


def test_production_pool_accepts_injected_jsonl_metrics(tmp_path):
    good = "ACGU" * 5
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        '{"dataset_index": 0, "status": "ok", "rmsd": 1.0, "tm_score": 0.6, "gdt": 0.7}\n',
        encoding="utf-8",
    )
    processed = {good: raw_record(good)}
    split = {"train": [0], "validation": [], "test": []}

    manifest = build_target_pool(
        processed,
        split,
        TargetPoolConfig(limit=1, calibration_mode="jsonl", baseline_mode="jsonl"),
        JsonlMetricsBackend(metrics_path, kind="calibration"),
        JsonlMetricsBackend(metrics_path, kind="baseline"),
    )

    assert manifest["summary"]["selected"] == 1
    assert manifest["targets"][0]["calibration"]["backend"] == "jsonl:calibration"


def test_materialize_condition_featurizes_on_cpu_then_moves_to_requested_device(tmp_path):
    good = "ACGU" * 5
    manifest = build_target_pool(
        {"demo": raw_record(good)},
        {"train": [0], "validation": [], "test": []},
        TargetPoolConfig(limit=1, dev_mode=True),
        FakeCalibration(),
        FakeBaseline(),
    )
    manifest_path = tmp_path / "targets.pt"
    write_manifest(manifest, manifest_path)
    target = load_target_pool(manifest_path)[0]
    requested = "cuda" if torch.cuda.is_available() else "cpu"

    condition = materialize_condition(target, device=requested)

    assert condition.node_features.device.type == requested
    assert condition.edge_index.device.type == requested
    assert condition.pyg_data.seq.device.type == requested
    condition.validate()


def test_materialize_condition_uses_selected_manifest_conformer_not_raw_conformer_zero(tmp_path):
    sequence = "ACGU" * 5
    invalid_first = full_atom_coords(len(sequence))
    invalid_first[0, 0] = torch.tensor([1.0e-5, 1.0e-5, 1.0e-5])
    valid_second = full_atom_coords(len(sequence))
    raw = {
        "sequence": sequence,
        "id_list": ["bad_conformer", "selected_conformer"],
        "coords_list": [invalid_first, valid_second],
    }
    manifest = build_target_pool(
        {"demo": raw},
        {"train": [0], "validation": [], "test": []},
        TargetPoolConfig(limit=1, dev_mode=True),
        FakeCalibration(),
        FakeBaseline(),
    )
    assert manifest["targets"][0]["metadata"]["conformer_index"] == 1
    manifest_path = tmp_path / "targets.pt"
    write_manifest(manifest, manifest_path)
    target = load_target_pool(manifest_path)[0]

    condition = materialize_condition(target, device="cpu")

    assert condition.pyg_data.seq.shape[0] == len(sequence)
    assert condition.node_features.shape[0] == len(sequence)
    assert target.reference_c4p_coords.shape[0] == len(sequence)
    assert condition.pyg_data.seq.shape[0] == target.reference_c4p_coords.shape[0]
