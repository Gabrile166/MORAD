from __future__ import annotations

import torch

from src.constants import RNA_ATOMS
from src.evaluation.targets import build_evaluation_target_pool


def test_evaluation_pool_uses_explicit_held_out_position_and_is_marked_nontraining() -> None:
    coords = torch.ones((4, len(RNA_ATOMS), 3), dtype=torch.float32)
    processed = {
        "train": {"sequence": "AUGC", "coords_list": [coords], "id_list": ["train"]},
        "heldout": {
            "sequence": "AUGC",
            "coords_list": [coords],
            "id_list": ["heldout"],
            "sec_struct_list": ["...."],
            "rfam_list": ["test-family"],
        },
    }

    manifest = build_evaluation_target_pool(processed, {"test": [1]}, test_ids=[0])

    assert manifest["evaluation_only"] is True
    assert manifest["training_use_forbidden"] is True
    assert manifest["targets"][0]["metadata"]["dataset_index"] == 1
    assert manifest["targets"][0]["metadata"]["evaluation_test_id"] == 0
