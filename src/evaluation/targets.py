"""Evaluation-only target manifests for explicit held-out DAS test examples."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from collections import Counter
from typing import Any, Mapping, Sequence

import torch

from src.constants import FILL_VALUE, RNA_ATOMS, RNA_NUCLEOTIDES
from src.data.data_utils import get_backbone_coords, get_c4p_coords
from src.rl.targets import (
    SCHEMA_VERSION,
    RLTarget,
    TargetPoolConfig,
    make_target,
    stable_target_id,
    structure_digest,
)


def build_evaluation_target_pool(
    processed_data: Mapping[str, Mapping[str, Any]],
    split_indices: Mapping[str, Sequence[int]],
    *,
    test_ids: Sequence[int],
    split_name: str = "das",
    impute_missing_coordinates: bool = False,
    skip_invalid: bool = False,
) -> dict[str, Any]:
    """Build a manifest that is valid for scoring but explicitly forbidden for training.

    ``test_ids`` are positions inside the held-out test split, not global dataset
    indices. This function deliberately lives outside ``src.rl.targets`` so the
    training-pool leakage guard remains unchanged.
    """

    keys = list(processed_data)
    test_indices = [int(index) for index in split_indices.get("test", ())]
    config = TargetPoolConfig(
        split_name=split_name,
        source_split="test",
        min_length_exclusive=0,
        max_length=10_000,
        require_calibrated=False,
        calibration_mode="none",
        baseline_mode="none",
    )
    targets: list[dict[str, Any]] = []
    selected: list[dict[str, int]] = []
    rejected: Counter[str] = Counter()
    imputed_target_count = 0
    imputed_residue_count = 0
    for test_id in test_ids:
        if test_id < 0 or test_id >= len(test_indices):
            raise IndexError(f"test-id {test_id} out of range (size={len(test_indices)})")
        dataset_index = test_indices[test_id]
        if dataset_index < 0 or dataset_index >= len(keys):
            raise IndexError(f"global dataset index {dataset_index} out of range")
        raw = processed_data[keys[dataset_index]]
        imputation: dict[str, Any] = {"imputed_residue_count": 0, "strict_complete": True}
        if impute_missing_coordinates:
            target, reason, imputation = _make_evaluation_target_with_imputation(raw, dataset_index, config)
        else:
            target, reason = make_target(raw, dataset_index, config)
        if target is None:
            if skip_invalid:
                rejected[reason or "unknown"] += 1
                continue
            raise ValueError(f"test-id {test_id} cannot form an evaluation target: {reason}")
        metadata = target.metadata()
        metadata["evaluation_test_id"] = int(test_id)
        metadata.update(imputation)
        if int(imputation["imputed_residue_count"]) > 0:
            imputed_target_count += 1
            imputed_residue_count += int(imputation["imputed_residue_count"])
        evaluation_raw = dict(raw)
        evaluation_raw["_evaluation"] = {
            "evaluation_test_id": int(test_id),
            **imputation,
        }
        targets.append(
            {
                "metadata": metadata,
                "raw_record": evaluation_raw,
                "ref_backbone_coords": target.ref_backbone_coords,
                "ref_c4p_coords": target.ref_c4p_coords,
                "mask_coords": target.mask_coords,
                "calibration": {"status": "skipped", "backend": "evaluation-only"},
                "frozen_baseline": {"status": "skipped", "backend": "evaluation-only"},
            }
        )
        selected.append({"test_id": int(test_id), "dataset_index": int(dataset_index)})

    if not targets:
        raise ValueError("At least one held-out test-id is required")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True,
        "training_use_forbidden": True,
        "resolved_config": {
            **asdict(config),
            "test_ids": [int(value) for value in test_ids],
            "impute_missing_coordinates": bool(impute_missing_coordinates),
            "skip_invalid": bool(skip_invalid),
        },
        "summary": {
            "total_records": len(processed_data),
            "candidate_indices": len(test_ids),
            "selected": len(targets),
            "rejected": dict(sorted(rejected.items())),
            "strict_complete_targets": len(targets) - imputed_target_count,
            "imputed_targets": imputed_target_count,
            "imputed_residues": imputed_residue_count,
            "selected_test_indices": selected,
        },
        "targets": targets,
    }


def impute_required_atom_coordinates(coords: torch.Tensor, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly fill only the atom channels required by RIDE/RiboDiffusion.

    Returns the imputed all-atom tensor and a residue mask marking positions
    whose RIDE P/C4′/base-N condition was originally complete.
    """

    coords = torch.as_tensor(coords, dtype=torch.float32).clone()
    if coords.shape != (len(sequence), len(RNA_ATOMS), 3):
        raise ValueError(f"unexpected coordinate shape {tuple(coords.shape)}")
    atom_index = {name: index for index, name in enumerate(RNA_ATOMS)}
    selected_base = torch.stack(
        [
            coords[index, atom_index["N9" if base in {"A", "G"} else "N1"]]
            for index, base in enumerate(sequence)
        ]
    )
    ride_channels = torch.stack(
        [coords[:, atom_index["P"]], coords[:, atom_index["C4'"]], selected_base],
        dim=1,
    )
    original_complete = ~_missing_rows(ride_channels)

    for atom_name in ("P", "C4'", "C1'"):
        index = atom_index[atom_name]
        coords[:, index] = _interpolate_missing(coords[:, index])
    selected_base = _interpolate_missing(selected_base)
    for residue_index, base in enumerate(sequence):
        coords[residue_index, atom_index["N9" if base in {"A", "G"} else "N1"]] = selected_base[residue_index]
    return coords, original_complete


def _make_evaluation_target_with_imputation(
    raw: Mapping[str, Any],
    dataset_index: int,
    config: TargetPoolConfig,
) -> tuple[RLTarget | None, str | None, dict[str, Any]]:
    sequence = str(raw.get("sequence", ""))
    if not sequence or any(base not in set(RNA_NUCLEOTIDES) for base in sequence):
        return None, "nonstandard_base", {}
    candidates: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
    for conformer_index, value in enumerate(raw.get("coords_list") or []):
        try:
            imputed, original_complete = impute_required_atom_coordinates(torch.as_tensor(value), sequence)
        except ValueError:
            continue
        candidates.append((int((~original_complete).sum().item()), conformer_index, imputed, original_complete))
    if not candidates:
        return None, "no_valid_conformer", {}
    missing_count, conformer_index, imputed, original_complete = min(candidates, key=lambda item: item[:2])
    backbone = get_backbone_coords(imputed, sequence).float().contiguous()
    c4p = get_c4p_coords(backbone).float().contiguous()
    pdb_ids = raw.get("id_list") or []
    pdb_id = str(pdb_ids[conformer_index]) if conformer_index < len(pdb_ids) else f"conformer_{conformer_index}"
    digest = structure_digest(sequence, c4p, config.structure_hash_decimals)
    target = RLTarget(
        target_id=stable_target_id(config.source_split, dataset_index, sequence, digest),
        dataset_index=int(dataset_index),
        source_split=config.source_split,
        sequence=sequence,
        length=len(sequence),
        pdb_id=pdb_id,
        conformer_index=conformer_index,
        structure_hash=digest,
        ref_backbone_coords=backbone,
        ref_c4p_coords=c4p,
        mask_coords=original_complete.contiguous(),
    )
    return target, None, {
        "imputed_residue_count": missing_count,
        "strict_complete": missing_count == 0,
        "coordinate_imputation": "linear_nearest_required_atoms_v1" if missing_count else "none",
    }


def _missing_rows(coords: torch.Tensor) -> torch.Tensor:
    flat = coords.reshape(coords.shape[0], -1)
    return (~torch.isfinite(flat)).any(dim=1) | torch.isclose(flat, torch.tensor(FILL_VALUE)).any(dim=1)


def _interpolate_missing(coords: torch.Tensor) -> torch.Tensor:
    coords = coords.clone()
    missing = _missing_rows(coords)
    valid_indices = torch.nonzero(~missing, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("required atom channel has no valid coordinates")
    for index in torch.nonzero(missing, as_tuple=False).flatten().tolist():
        left = valid_indices[valid_indices < index]
        right = valid_indices[valid_indices > index]
        if left.numel() and right.numel():
            lo = int(left[-1])
            hi = int(right[0])
            weight = float(index - lo) / float(hi - lo)
            coords[index] = coords[lo] * (1.0 - weight) + coords[hi] * weight
        elif left.numel():
            coords[index] = coords[int(left[-1])]
        else:
            coords[index] = coords[int(right[0])]
    return coords
