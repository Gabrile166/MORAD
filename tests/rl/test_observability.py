from __future__ import annotations

import json

from src.rl.observability import build_run_manifest, json_safe, sha256_file


class BadItem:
    def item(self):
        raise RuntimeError("cannot scalarize")


def test_json_safe_preserves_error_evidence_for_failed_item_conversion():
    payload = json_safe({"value": BadItem()})

    json.dumps(payload)
    assert "BadItem" in payload["value"]["repr"]
    assert "cannot scalarize" in payload["value"]["json_error"]


def test_run_manifest_records_artifact_hashes_and_reference_repositories(tmp_path):
    repo = tmp_path / "RIDER"
    repo.mkdir()
    pool = repo / "pool.pt"
    ckpt = repo / "saved_models" / "checkpoint.h5"
    rhofold = tmp_path / "third_party" / "rhofold_protocol" / "checkpoints" / "rhofold_pretrained_params.pt"
    pool.write_bytes(b"pool")
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"ride")
    rhofold.parent.mkdir(parents=True)
    rhofold.write_bytes(b"rhofold")
    (tmp_path / "third_party" / "DiffusionNFT" / ".git").mkdir(parents=True)
    (tmp_path / "third_party" / "verl" / ".git").mkdir(parents=True)
    (tmp_path / "third_party" / "RhoFold" / ".git").mkdir(parents=True)
    (tmp_path / "third_party" / "rhofold_protocol" / ".git").mkdir(parents=True)

    manifest = build_run_manifest(
        repo,
        {
            "data": {
                "pool_manifest": "pool.pt",
                "ride_checkpoint": "saved_models/checkpoint.h5",
            },
            "reward": {
                "oracle": {
                    "ckpt": "../third_party/rhofold_protocol/checkpoints/rhofold_pretrained_params.pt",
                }
            },
        },
    )

    assert manifest["artifact_hashes"]["pool_manifest"]["sha256"] == sha256_file(pool)
    assert manifest["artifact_hashes"]["ride_checkpoint"]["sha256"] == sha256_file(ckpt)
    assert manifest["artifact_hashes"]["rhofold_checkpoint"]["sha256"] == sha256_file(rhofold)
    assert set(manifest["reference_repositories"]) == {
        "DiffusionNFT",
        "verl",
        "RhoFold",
        "rhofold_protocol",
        "RiboDiffusion",
        "USalign",
    }
    assert manifest["reference_repositories"]["DiffusionNFT"]["exists"] is True
    assert manifest["reference_repositories"]["DiffusionNFT"]["is_git"] is True
