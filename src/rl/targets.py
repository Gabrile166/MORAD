"""RL target-pool construction for RIDE fine-tuning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import torch

from src.constants import FILL_VALUE, RNA_ATOMS, RNA_NUCLEOTIDES
from src.data.data_utils import get_backbone_coords, get_c4p_coords
from src.data.featurizer import RNAGraphFeaturizer
from src.rl.protocol import ConditionBundle, TargetRecord


SCHEMA_VERSION = "ride_rl_target_pool.v1"
VALID_CALIBRATION_STATUSES = {"ok", "skipped"}
VALID_RNA_BASES = set(RNA_NUCLEOTIDES)


@dataclass(frozen=True)
class TargetPoolConfig:
    """Configuration used to select stable RL targets from processed.pt."""

    split_name: str = "das"
    source_split: str = "train"
    min_length_exclusive: int = 18
    max_length: int = 512
    limit: Optional[int] = None
    require_calibrated: bool = True
    calibration_mode: str = "native-reference"
    baseline_mode: str = "native-sequence"
    calibration_jsonl: Optional[str] = None
    baseline_jsonl: Optional[str] = None
    structure_hash_decimals: int = 3
    dev_mode: bool = False


@dataclass(frozen=True)
class RLTarget:
    """One immutable RNA condition target used by rollout/reward/trainer code."""

    target_id: str
    dataset_index: int
    source_split: str
    sequence: str
    length: int
    pdb_id: str
    conformer_index: int
    structure_hash: str
    ref_backbone_coords: torch.Tensor
    ref_c4p_coords: torch.Tensor
    mask_coords: torch.Tensor

    def metadata(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "dataset_index": self.dataset_index,
            "source_split": self.source_split,
            "sequence": self.sequence,
            "length": self.length,
            "pdb_id": self.pdb_id,
            "conformer_index": self.conformer_index,
            "structure_hash": self.structure_hash,
        }


@dataclass(frozen=True)
class TargetBuildStats:
    total_records: int
    candidate_indices: int
    selected: int
    rejected: Mapping[str, int]


class CalibrationBackend(Protocol):
    name: str

    def evaluate(self, target: RLTarget) -> Mapping[str, Any]:
        ...


class BaselineBackend(Protocol):
    name: str

    def evaluate(self, target: RLTarget) -> Mapping[str, Any]:
        ...


class NativeReferenceCalibration:
    """Cheap deterministic calibration for wiring tests and dev manifests.

    This does not claim RhoFold oracle quality. It records the native structure
    as the reference anchor so selected targets have non-pending calibration
    status and stable reward-shape fields before the expensive oracle pass exists.
    """

    name = "native-reference"

    def evaluate(self, target: RLTarget) -> Mapping[str, Any]:
        return {
            "status": "ok",
            "backend": self.name,
            "oracle": "native_reference",
            "rmsd": 0.0,
            "tm_score": 1.0,
            "gdt": 1.0,
            "note": "native structure self-check; replace with RhoFold+ calibration for production pools",
        }


class NullCalibration:
    name = "none"

    def evaluate(self, target: RLTarget) -> Mapping[str, Any]:
        return {"status": "uncalibrated", "backend": self.name}


class NativeSequenceBaseline:
    name = "native-sequence"

    def evaluate(self, target: RLTarget) -> Mapping[str, Any]:
        return {
            "status": "ok",
            "backend": self.name,
            "sequence": target.sequence,
            "sequence_recovery": 1.0,
        }


class NullBaseline:
    name = "none"

    def evaluate(self, target: RLTarget) -> Mapping[str, Any]:
        return {"status": "skipped", "backend": self.name}


class JsonlMetricsBackend:
    """Load precomputed production calibration/baseline metrics by stable target identity."""

    name = "jsonl"

    def __init__(self, path: Path | str, *, kind: str) -> None:
        self.path = Path(path)
        self.kind = kind
        self.records = self._load(self.path)

    def evaluate(self, target: RLTarget) -> Mapping[str, Any]:
        for key in (target.target_id, str(target.dataset_index), target.sequence):
            if key in self.records:
                record = dict(self.records[key])
                record.setdefault("status", "ok")
                record.setdefault("backend", f"jsonl:{self.kind}")
                return record
        return {"status": "missing", "backend": f"jsonl:{self.kind}", "path": str(self.path)}

    @staticmethod
    def _load(path: Path) -> Dict[str, Mapping[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(path)
        records: Dict[str, Mapping[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                keys = [
                    payload.get("target_id"),
                    payload.get("dataset_index"),
                    payload.get("sequence"),
                ]
                for key in keys:
                    if key is not None:
                        records[str(key)] = payload
        return records


def load_processed_data(data_path: Path | str) -> "Dict[str, Mapping[str, Any]]":
    data = torch.load(Path(data_path), map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Expected processed data dict, got {type(data)!r}")
    return data


def load_split_indices(split_path: Path | str) -> Dict[str, List[int]]:
    split = torch.load(Path(split_path), map_location="cpu", weights_only=False)
    if isinstance(split, Mapping):
        return {str(k): [int(i) for i in v] for k, v in split.items()}
    if isinstance(split, tuple) and len(split) == 3:
        return {
            "train": [int(i) for i in split[0]],
            "validation": [int(i) for i in split[1]],
            "test": [int(i) for i in split[2]],
        }
    raise TypeError(f"Unsupported split file format: {type(split)!r}")


def build_target_pool(
    processed_data: Mapping[str, Mapping[str, Any]],
    split_indices: Mapping[str, Sequence[int]],
    config: TargetPoolConfig,
    calibration_backend: CalibrationBackend,
    baseline_backend: BaselineBackend,
) -> Dict[str, Any]:
    _validate_pool_backend_modes(config)
    keys = list(processed_data.keys())
    selected_indices = _source_indices(split_indices, config.source_split)
    test_indices = set(int(i) for i in split_indices.get("test", []))
    rejected: Dict[str, int] = {}
    targets: List[Dict[str, Any]] = []

    for dataset_index in selected_indices:
        if config.limit is not None and len(targets) >= config.limit:
            break
        if dataset_index in test_indices:
            _bump(rejected, "test_leakage")
            continue
        if dataset_index < 0 or dataset_index >= len(keys):
            _bump(rejected, "index_out_of_range")
            continue

        raw = processed_data[keys[dataset_index]]
        target, reason = make_target(raw, dataset_index, config)
        if target is None:
            _bump(rejected, reason or "unknown")
            continue

        calibration = dict(calibration_backend.evaluate(target))
        baseline = dict(baseline_backend.evaluate(target))
        if config.require_calibrated and not _is_calibrated(calibration):
            _bump(rejected, f"calibration_{calibration.get('status', 'missing')}")
            continue

        targets.append(
            {
                "metadata": target.metadata(),
                "raw_record": raw,
                "ref_backbone_coords": target.ref_backbone_coords,
                "ref_c4p_coords": target.ref_c4p_coords,
                "mask_coords": target.mask_coords,
                "calibration": calibration,
                "frozen_baseline": baseline,
            }
        )

    stats = TargetBuildStats(
        total_records=len(processed_data),
        candidate_indices=len(selected_indices),
        selected=len(targets),
        rejected=dict(sorted(rejected.items())),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_config": asdict(config),
        "summary": asdict(stats),
        "targets": targets,
    }


def make_target(
    raw: Mapping[str, Any],
    dataset_index: int,
    config: TargetPoolConfig,
) -> Tuple[Optional[RLTarget], Optional[str]]:
    sequence = str(raw.get("sequence", ""))
    length = len(sequence)
    if length <= config.min_length_exclusive:
        return None, "too_short"
    if length > config.max_length:
        return None, "too_long"
    if any(base not in VALID_RNA_BASES for base in sequence):
        return None, "nonstandard_base"

    coords_list = raw.get("coords_list") or []
    id_list = raw.get("id_list") or []
    for conformer_index, coords in enumerate(coords_list):
        if not isinstance(coords, torch.Tensor):
            coords = torch.as_tensor(coords)
        if coords.shape[0] != length:
            continue
        backbone = get_backbone_coords(coords.float().cpu(), sequence)
        mask = _complete_backbone_mask(backbone)
        if not bool(mask.all()):
            continue
        if not bool(torch.isfinite(backbone).all()):
            continue
        c4p = get_c4p_coords(backbone)
        if not bool(torch.isfinite(c4p).all()):
            continue

        pdb_id = str(id_list[conformer_index]) if conformer_index < len(id_list) else f"conformer_{conformer_index}"
        structure_hash = structure_digest(sequence, c4p, config.structure_hash_decimals)
        target_id = stable_target_id(config.source_split, dataset_index, sequence, structure_hash)
        return (
            RLTarget(
                target_id=target_id,
                dataset_index=int(dataset_index),
                source_split=config.source_split,
                sequence=sequence,
                length=length,
                pdb_id=pdb_id,
                conformer_index=int(conformer_index),
                structure_hash=structure_hash,
                ref_backbone_coords=backbone.contiguous(),
                ref_c4p_coords=c4p.contiguous(),
                mask_coords=mask.contiguous(),
            ),
            None,
        )
    return None, "no_valid_conformer"


def write_manifest(manifest: Mapping[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(dict(manifest), tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_summary_json(manifest: Mapping[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    summary = {
        "schema_version": manifest["schema_version"],
        "created_at_utc": manifest["created_at_utc"],
        "resolved_config": manifest["resolved_config"],
        "summary": manifest["summary"],
        "targets": [target["metadata"] for target in manifest["targets"]],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_target_pool(path: Path | str) -> List[TargetRecord]:
    """Load a torch-saved target-pool manifest into typed records."""

    if path is None:
        raise ValueError("data.pool_manifest is required")
    manifest_path = Path(path)
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported target pool manifest: {manifest_path}")
    records: List[TargetRecord] = []
    version = str(manifest.get("created_at_utc", "unknown"))
    for item in manifest.get("targets", []):
        metadata = item["metadata"]
        calibration = item.get("calibration", {})
        baseline = item.get("frozen_baseline", {})
        backbone = torch.as_tensor(item["ref_backbone_coords"], dtype=torch.float32)
        c4p = torch.as_tensor(item["ref_c4p_coords"], dtype=torch.float32)
        conformer_index = int(metadata.get("conformer_index", 0))
        c1p = _reference_c1p(item.get("raw_record"), conformer_index, int(metadata["length"]))
        record = TargetRecord(
            target_id=str(metadata["target_id"]),
            sequence_native=str(metadata["sequence"]),
            length=int(metadata["length"]),
            structure_ref=f"tensor://{metadata['target_id']}/ref_c4p_coords",
            structure_hash=str(metadata["structure_hash"]),
            split=str(metadata.get("source_split", "train")),
            source=str(metadata.get("pdb_id", "processed.pt")),
            dataset_version=version,
            native_oracle_metrics=_metrics_from_mapping(calibration),
            frozen_ride_metrics=_metrics_from_mapping(baseline),
            calibration_status=str(calibration.get("status", "unknown")),
            reference_backbone_coords=backbone,
            reference_c4p_coords=c4p,
            reference_c1p_coords=c1p,
            raw_record=item.get("raw_record"),
            dataset_index=int(metadata.get("dataset_index", -1)),
            conformer_index=conformer_index,
        ).validate()
        records.append(record)
    if not records:
        raise ValueError(f"Target pool manifest contains no targets: {manifest_path}")
    return records


def _reference_c1p(raw_record: Any, conformer_index: int, length: int) -> torch.Tensor | None:
    if not isinstance(raw_record, Mapping):
        return None
    coords_list = raw_record.get("coords_list") or []
    if conformer_index < 0 or conformer_index >= len(coords_list):
        return None
    coords = torch.as_tensor(coords_list[conformer_index], dtype=torch.float32)
    atom_index = RNA_ATOMS.index("C1'")
    if coords.ndim != 3 or coords.shape[0] != length or coords.shape[1] <= atom_index:
        return None
    c1p = coords[:, atom_index, :].contiguous()
    if not torch.isfinite(c1p).all() or bool((c1p == FILL_VALUE).all(dim=1).any()):
        return None
    return c1p


def materialize_condition(
    target: TargetRecord,
    *,
    condition_noise_scale: float = 0.0,
    device: str | torch.device = "cpu",
    round_id: int = 0,
) -> ConditionBundle:
    """Build a validated condition bundle using RIDE's existing graph featurizer."""

    if condition_noise_scale != 0.0:
        raise ValueError("POC materialize_condition requires condition_noise_scale=0")
    if target.reference_backbone_coords is None:
        raise ValueError("TargetRecord.reference_backbone_coords is required to materialize a RIDE condition")
    raw_record = {
        "sequence": target.sequence_native,
        "id_list": [target.source],
        "coords_list": [target.reference_backbone_coords.detach().cpu()],
    }
    featurizer = RNAGraphFeaturizer(split="validation", max_num_conformers=1, noise_scale=0.0, device="cpu")
    graph = featurizer(raw_record)
    if int(graph.seq.shape[0]) != target.length:
        raise ValueError(
            f"Condition node count {int(graph.seq.shape[0])} does not match target length {target.length} "
            f"for {target.target_id}"
        )
    node_features = torch.cat(
        [
            graph.node_s.reshape(graph.node_s.shape[0], -1),
            graph.node_v.reshape(graph.node_v.shape[0], -1),
        ],
        dim=-1,
    ).detach()
    edge_features = torch.cat(
        [
            graph.edge_s.reshape(graph.edge_s.shape[0], -1),
            graph.edge_v.reshape(graph.edge_v.shape[0], -1),
        ],
        dim=-1,
    ).detach()
    condition_id = f"{target.target_id}:round{round_id}:noise0"
    feature_hash = hashlib.sha256()
    feature_hash.update(target.structure_hash.encode("utf-8"))
    feature_hash.update(str(tuple(graph.seq.detach().cpu().tolist())).encode("utf-8"))
    bundle = ConditionBundle(
        condition_id=condition_id,
        target_id=target.target_id,
        round_id=int(round_id),
        node_features=node_features,
        edge_features=edge_features.detach(),
        edge_index=graph.edge_index.detach(),
        node_mask=torch.ones(int(graph.seq.shape[0]), dtype=torch.bool, device=graph.seq.device),
        condition_noise_scale=0.0,
        feature_hash=feature_hash.hexdigest(),
        pyg_data=graph,
    ).validate()
    return bundle.to(device).validate()


def calibration_backend_from_name(name: str, jsonl_path: str | Path | None = None) -> CalibrationBackend:
    if name == "native-reference":
        return NativeReferenceCalibration()
    if name == "jsonl":
        if jsonl_path is None:
            raise ValueError("calibration_mode=jsonl requires --calibration-jsonl")
        return JsonlMetricsBackend(jsonl_path, kind="calibration")
    if name == "none":
        return NullCalibration()
    raise ValueError(f"Unknown calibration backend: {name}")


def baseline_backend_from_name(name: str, jsonl_path: str | Path | None = None) -> BaselineBackend:
    if name == "native-sequence":
        return NativeSequenceBaseline()
    if name == "jsonl":
        if jsonl_path is None:
            raise ValueError("baseline_mode=jsonl requires --baseline-jsonl")
        return JsonlMetricsBackend(jsonl_path, kind="baseline")
    if name == "none":
        return NullBaseline()
    raise ValueError(f"Unknown baseline backend: {name}")


def _validate_pool_backend_modes(config: TargetPoolConfig) -> None:
    placeholder = config.calibration_mode == "native-reference" or config.baseline_mode == "native-sequence"
    if placeholder and not config.dev_mode:
        raise ValueError(
            "native-reference/native-sequence are dev placeholders. "
            "Pass dev_mode=True/--dev-mode, or use real RhoFold+ calibration and frozen RIDE baseline for production."
        )


def _metrics_from_mapping(value: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for source_key, dest_key in (("rmsd", "rmsd"), ("tm_score", "tm_score"), ("gdt", "gdt_ts"), ("gdt_ts", "gdt_ts")):
        if source_key in value:
            out[dest_key] = float(value[source_key])
    return out


def _source_indices(split_indices: Mapping[str, Sequence[int]], source_split: str) -> List[int]:
    if source_split not in split_indices:
        raise KeyError(f"Split `{source_split}` not found. Available: {sorted(split_indices)}")
    return [int(i) for i in split_indices[source_split]]


def _complete_backbone_mask(backbone: torch.Tensor) -> torch.Tensor:
    flat = backbone.reshape(backbone.shape[0], -1)
    finite = torch.isfinite(flat).all(dim=1)
    not_fill = (flat != FILL_VALUE).all(dim=1)
    return finite & not_fill


def _is_calibrated(calibration: Mapping[str, Any]) -> bool:
    status = calibration.get("status")
    if status not in VALID_CALIBRATION_STATUSES:
        return False
    for key in ("rmsd", "tm_score", "gdt"):
        if key in calibration:
            value = float(calibration[key])
            if not math.isfinite(value):
                return False
    return True


def _bump(counts: Dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def stable_target_id(source_split: str, dataset_index: int, sequence: str, structure_hash: str) -> str:
    digest = hashlib.sha256(f"{source_split}|{dataset_index}|{sequence}|{structure_hash}".encode("utf-8")).hexdigest()
    return f"ride-{source_split}-{dataset_index:05d}-{digest[:12]}"


def structure_digest(sequence: str, c4p_coords: torch.Tensor, decimals: int = 3) -> str:
    rounded = torch.round(c4p_coords.detach().cpu().float() * (10**decimals)).to(torch.int32)
    digest = hashlib.sha256()
    digest.update(sequence.encode("utf-8"))
    digest.update(rounded.numpy().tobytes())
    return digest.hexdigest()
