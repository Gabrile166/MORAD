from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from Bio import Align
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from src.constants import FILL_VALUE, RNA_ATOMS
from src.data.data_utils import get_backbone_coords, get_c4p_coords
from src.rl.targets import SCHEMA_VERSION, structure_digest, write_manifest


CANONICAL_BASES = {"A", "U", "G", "C"}
COMMON_MODIFIED_BASE_PARENTS = {
    "16B": "C",
    "1MA": "A",
    "1MG": "G",
    "1RN": "U",
    "2MG": "G",
    "2MU": "U",
    "3AU": "U",
    "4OC": "C",
    "4SU": "U",
    "5BU": "U",
    "5MC": "C",
    "5MU": "U",
    "6IA": "A",
    "70U": "U",
    "73W": "C",
    "7MG": "G",
    "7OK": "C",
    "8AN": "A",
    "A23": "A",
    "A2M": "A",
    "AET": "A",
    "CBV": "C",
    "CCC": "C",
    "GDP": "G",
    "GTP": "G",
    "H2U": "U",
    "M3X": "C",
    "MIA": "A",
    "MNU": "U",
    "OMC": "C",
    "OMG": "G",
    "OMU": "U",
    "PGP": "G",
    "PPU": "A",
    "PSU": "U",
    "QUO": "G",
    "RSP": "C",
    "RSQ": "C",
    "T6A": "A",
    "U23": "U",
    "U8U": "U",
    "YG": "G",
}
MODIFIED_ATOM_ALIASES = {
    "GDP": {"PA": "P"},
    "GTP": {"PA": "P"},
}
DEFAULT_RELEASE_DATE_CUTOFF = "2025-01-01"
_LCS_ALIGNER = Align.PairwiseAligner()
_LCS_ALIGNER.mode = "global"
_LCS_ALIGNER.match_score = 1.0
_LCS_ALIGNER.mismatch_score = 0.0
_LCS_ALIGNER.open_gap_score = 0.0
_LCS_ALIGNER.extend_gap_score = 0.0


@dataclass(frozen=True)
class RNA3DBCandidate:
    pdb_id: str
    auth_chain_id: str
    component: str
    cluster_id: str
    sequence: str
    release_date: str
    resolution: float
    metadata: Mapping[str, Any]

    @property
    def pdb_chain(self) -> str:
        return normalize_pdb_chain(self.pdb_id, self.auth_chain_id)

    def quality_key(self) -> tuple[float, int, str, str]:
        return (self.resolution, self.length_delta(), self.pdb_id, self.auth_chain_id)

    def length_delta(self) -> int:
        midpoint = (32 + 280) // 2
        return abs(len(self.sequence) - midpoint)


@dataclass(frozen=True)
class BuildRNA3DBConfig:
    release_date_cutoff: str = DEFAULT_RELEASE_DATE_CUTOFF
    min_length: int = 32
    max_length: int = 280
    max_resolution: float = 4.0
    component_cap: int = 25
    limit: int | None = None
    overfetch: int = 3
    source_split: str = "rna3db_train"
    homology_screen: bool = True
    homology_threshold: float = 0.80
    homology_kmer: int = 6
    max_mmcif_bytes: int = 256 * 1024 * 1024
    max_terminal_trim: int = 10
    max_terminal_trim_ratio: float = 0.0
    max_internal_gap_ratio: float = 0.0


FetchMmCif = Callable[[str, Path, int], None]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_train_candidates(split_payload: Mapping[str, Any]) -> list[RNA3DBCandidate]:
    train = split_payload.get("train_set", split_payload)
    candidates: list[RNA3DBCandidate] = []
    for component_hint, component in _named_items(_value_for_any_key(train, ("component", "components")) or train):
        if not isinstance(component, Mapping):
            continue
        component_id = str(
            _value_for_any_key(component, ("component", "component_id", "name", "id"))
            or component_hint
            or "unknown_component"
        )
        for cluster_hint, cluster in _cluster_items(component):
            cluster_id = str(_value_for_any_key(cluster, ("cluster_id", "cluster", "id", "name")) or cluster_hint)
            for chain_hint, chain in _named_items(_value_for_any_key(cluster, ("chains", "chain", "members", "entries")) or cluster):
                if not isinstance(chain, Mapping):
                    continue
                candidate = _candidate_from_chain(chain, component_id, cluster_id, chain_hint)
                if candidate is not None:
                    candidates.append(candidate)
    return candidates


def select_rna3db_targets(
    split_payload: Mapping[str, Any],
    *,
    config: BuildRNA3DBConfig,
    output_dir: Path,
    cache_dir: Path,
    processed_sequences: Iterable[str] = (),
    ribodiffusion_exposed_chains: Iterable[str] = (),
    fetch_mmcif: FetchMmCif | None = None,
    mmcif_root: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = output_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    processed = {normalize_sequence(sequence) for sequence in processed_sequences if normalize_sequence(sequence)}
    homology_index = NearHomologyIndex(processed, threshold=config.homology_threshold, k=config.homology_kmer)
    exposed = {normalize_pdb_chain_text(value) for value in ribodiffusion_exposed_chains}
    local_mmcifs = index_local_chain_mmcifs(mmcif_root) if mmcif_root is not None else {}
    rejected: Counter[str] = Counter()
    rejection_records: list[dict[str, Any]] = []
    candidates = parse_train_candidates(split_payload)

    by_cluster: dict[str, list[RNA3DBCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_cluster[candidate.cluster_id].append(candidate)

    eligible_by_cluster: dict[str, list[RNA3DBCandidate]] = {}
    cutoff = date.fromisoformat(config.release_date_cutoff)
    for cluster_id, cluster_candidates in by_cluster.items():
        cluster_chains = {candidate.pdb_chain for candidate in cluster_candidates}
        if cluster_chains & exposed:
            _reject_cluster(rejected, rejection_records, cluster_candidates, "ribodiffusion_cluster_exposed")
            continue
        eligible: list[RNA3DBCandidate] = []
        for candidate in cluster_candidates:
            reason = _metadata_rejection(candidate, config, cutoff)
            if reason is not None:
                _reject(rejected, rejection_records, candidate, reason)
                continue
            if candidate.sequence in processed:
                _reject(rejected, rejection_records, candidate, "processed_exact_sequence")
                continue
            eligible.append(candidate)
        if eligible:
            eligible_by_cluster[cluster_id] = sorted(eligible, key=lambda item: item.quality_key())

    cluster_representatives = [items[0] for items in eligible_by_cluster.values()]
    capped_clusters: list[RNA3DBCandidate] = []
    by_component: dict[str, list[RNA3DBCandidate]] = defaultdict(list)
    for candidate in cluster_representatives:
        by_component[candidate.component].append(candidate)
    for component, items in by_component.items():
        kept = sorted(items, key=lambda item: item.quality_key())[: config.component_cap]
        capped_clusters.extend(kept)
        for rejected_representative in sorted(items, key=lambda item: item.quality_key())[config.component_cap :]:
            _reject_cluster(
                rejected,
                rejection_records,
                eligible_by_cluster[rejected_representative.cluster_id],
                "component_cap",
            )

    desired = None if config.limit is None else max(config.limit * max(config.overfetch, 1), config.limit)
    shortlisted_clusters = sorted(capped_clusters, key=lambda item: item.quality_key())[:desired]
    shortlisted = [
        candidate
        for representative in shortlisted_clusters
        for candidate in eligible_by_cluster[representative.cluster_id]
    ]

    targets: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    selected_clusters: set[str] = set()
    for candidate in shortlisted:
        if candidate.cluster_id in selected_clusters:
            continue
        try:
            cif_path = local_mmcifs.get(candidate.pdb_chain)
            if cif_path is None:
                if mmcif_root is not None:
                    raise ValueError(f"local_mmcif_missing:{candidate.pdb_chain}")
                cif_path = download_mmcif(
                    candidate.pdb_id,
                    cache_dir,
                    fetch_mmcif=fetch_mmcif,
                    max_bytes=config.max_mmcif_bytes,
                )
            extracted = extract_auth_chain_from_mmcif(
                cif_path,
                candidate.auth_chain_id,
                max_internal_gap_ratio=config.max_internal_gap_ratio,
            )
        except ValueError as exc:
            _reject(rejected, rejection_records, candidate, str(exc))
            continue
        coordinate_trim_left = int(extracted.get("terminal_trim_left", 0))
        coordinate_trim_right = int(extracted.get("terminal_trim_right", 0))
        trim_left = coordinate_trim_left
        trim_right = coordinate_trim_right
        expected_after_trim = candidate.sequence[trim_left : len(candidate.sequence) - trim_right if trim_right else None]
        if extracted["sequence"] != expected_after_trim:
            substring_start = candidate.sequence.find(extracted["sequence"])
            substring_right = len(candidate.sequence) - substring_start - len(extracted["sequence"])
            trim_budget = max(
                config.max_terminal_trim,
                int(len(candidate.sequence) * config.max_terminal_trim_ratio),
            )
            gap_residues = int(extracted.get("internal_gap_residues", 0))
            if substring_start < 0 and gap_residues:
                # With interior gaps tolerated the extracted string is the metadata
                # sequence minus the unmodelled positions, so it is an order-preserving
                # subsequence rather than a contiguous substring and ``find`` cannot
                # locate it. Anchor on label_seq_id, which is the authoritative index
                # into the metadata sequence, and verify every observed residue agrees
                # with the base recorded there. That is a stricter check than ``find``:
                # it pins each residue to its own position instead of matching a run.
                label_ids = extracted.get("label_seq_ids") or []
                reference = candidate.sequence
                aligned = bool(label_ids) and all(
                    1 <= seq_id <= len(reference) and reference[seq_id - 1] == base
                    for seq_id, base in zip(label_ids, extracted["sequence"])
                )
                if aligned:
                    trim_left = label_ids[0] - 1
                    trim_right = len(reference) - label_ids[-1]
                    if trim_left + trim_right <= trim_budget:
                        substring_start = trim_left
                        substring_right = trim_right
            if substring_start < 0 or substring_start + substring_right > trim_budget:
                _reject(
                    rejected,
                    rejection_records,
                    candidate,
                    "metadata_sequence_mismatch",
                    extracted_sequence=extracted["sequence"],
                    terminal_trim_left=trim_left,
                    terminal_trim_right=trim_right,
                )
                continue
            trim_left = substring_start
            trim_right = substring_right
        if len(extracted["sequence"]) < config.min_length:
            _reject(rejected, rejection_records, candidate, "too_short_after_extract")
            continue
        if extracted["sequence"] in processed:
            _reject(rejected, rejection_records, candidate, "processed_exact_sequence_after_extract")
            continue
        if config.homology_screen:
            match = homology_index.find_near_match(extracted["sequence"])
            if match is not None:
                _reject(
                    rejected,
                    rejection_records,
                    candidate,
                    "processed_near_homology",
                    max_similarity=match["similarity"],
                    matched_sequence_hash=match["sequence_hash"],
                )
                continue

        if config.limit is not None and len(targets) >= config.limit:
            _reject(rejected, rejection_records, candidate, "limit_reached")
            continue

        coords = torch.as_tensor(extracted["coords"], dtype=torch.float32)
        backbone = get_backbone_coords(coords, extracted["sequence"]).float().contiguous()
        c4p = get_c4p_coords(backbone).float().contiguous()
        digest = structure_digest(extracted["sequence"], c4p)
        target_id = stable_rna3db_target_id(candidate, extracted["sequence"], digest)
        pdb_path = pdb_dir / f"{target_id}.pdb"
        write_single_chain_pdb(pdb_path, extracted["sequence"], coords)

        provenance = {
            "source": "RNA3DB",
            "source_split": "train_set",
            "component": candidate.component,
            "cluster_id": candidate.cluster_id,
            "pdb_id": candidate.pdb_id,
            "auth_chain_id": candidate.auth_chain_id,
            "pdb_chain": candidate.pdb_chain,
            "release_date": candidate.release_date,
            "resolution": candidate.resolution,
            "terminal_trim_left": trim_left,
            "terminal_trim_right": trim_right,
            "coordinate_terminal_trim_left": coordinate_trim_left,
            "coordinate_terminal_trim_right": coordinate_trim_right,
            "internal_gap_residues": int(extracted.get("internal_gap_residues", 0)),
            "internal_gap_ratio": float(extracted.get("internal_gap_ratio", 0.0)),
            "mmcif_path": str(cif_path),
            "pdb_path": str(pdb_path),
        }
        raw_record = {
            "sequence": extracted["sequence"],
            "id_list": [target_id],
            "coords_list": [coords],
            "_rna3db": provenance,
        }
        metadata = {
            "target_id": target_id,
            "dataset_index": len(targets),
            "source_split": config.source_split,
            "sequence": extracted["sequence"],
            "length": len(extracted["sequence"]),
            "pdb_id": target_id,
            "conformer_index": 0,
            "structure_hash": digest,
            "pdb_path": str(pdb_path),
            "rna3db": provenance,
        }
        targets.append(
            {
                "metadata": metadata,
                "raw_record": raw_record,
                "ref_backbone_coords": backbone,
                "ref_c4p_coords": c4p,
                "mask_coords": torch.ones(len(extracted["sequence"]), dtype=torch.bool),
                "calibration": {"status": "skipped", "backend": "rna3db-native-reference"},
                "frozen_baseline": {"status": "skipped", "backend": "rna3db-native-sequence"},
            }
        )
        selected_clusters.add(candidate.cluster_id)
        selection.append({**metadata, "raw_record_provenance": provenance})

    summary = {
        "total_chains": len(candidates),
        "candidate_clusters": len(by_cluster),
        "eligible_clusters": len(eligible_by_cluster),
        "shortlisted_clusters": len(shortlisted_clusters),
        "shortlisted_chains": len(shortlisted),
        "shortlisted": len(shortlisted),
        "selected": len(targets),
        "rejected": dict(sorted(rejected.items())),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_config": asdict(config),
        "summary": summary,
        "targets": targets,
        "selection": selection,
        "rejections": rejection_records,
    }


def write_rna3db_outputs(result: Mapping[str, Any], output_dir: Path, manifest_name: str = "rna3db_train_v1.pt") -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_name
    if manifest_path.suffix != ".pt":
        manifest_path = manifest_path.with_suffix(".pt")
    write_manifest(result, manifest_path)
    summary_path = manifest_path.with_suffix(".summary.json")
    selection_path = manifest_path.with_suffix(".selection.json")
    rejections_path = manifest_path.with_suffix(".rejections.json")
    summary_path.write_text(json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection_path.write_text(json.dumps(result["selection"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rejections_path.write_text(json.dumps(result["rejections"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "selection": selection_path,
        "rejections": rejections_path,
    }


def index_local_chain_mmcifs(root: Path) -> dict[str, Path]:
    """Index RNA3DB's official single-chain mmCIF release by PDB/auth-chain."""

    train_root = root / "train_set" if (root / "train_set").is_dir() else root
    index: dict[str, Path] = {}
    if not train_root.is_dir():
        raise FileNotFoundError(f"RNA3DB single-chain mmCIF root not found: {train_root}")
    for path in sorted(train_root.rglob("*.cif")):
        stem = path.stem
        if "_" not in stem:
            continue
        pdb_id, auth_chain_id = stem.split("_", 1)
        index.setdefault(normalize_pdb_chain(pdb_id, auth_chain_id), path)
    if not index:
        raise ValueError(f"RNA3DB single-chain mmCIF root contains no .cif files: {train_root}")
    return index


def download_mmcif(
    pdb_id: str,
    cache_dir: Path,
    *,
    fetch_mmcif: FetchMmCif | None = None,
    max_bytes: int = 256 * 1024 * 1024,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb = pdb_id.lower()
    path = cache_dir / f"{pdb}.cif"
    if path.exists() and 0 < path.stat().st_size <= max_bytes:
        return path
    if path.exists() and path.stat().st_size > max_bytes:
        raise ValueError(f"mmcif_too_large:{pdb_id}:{path.stat().st_size}")
    part = path.with_suffix(path.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0
    if resume_from > max_bytes:
        part.unlink()
        raise ValueError(f"mmcif_too_large:{pdb_id}:{resume_from}")
    if fetch_mmcif is not None:
        fetch_mmcif(pdb_id.upper(), part, resume_from)
        if part.stat().st_size > max_bytes:
            part.unlink()
            raise ValueError(f"mmcif_too_large:{pdb_id}")
    else:
        request = urllib.request.Request(f"https://files.rcsb.org/download/{pdb_id.upper()}.cif")
        if resume_from:
            request.add_header("Range", f"bytes={resume_from}-")
        with urllib.request.urlopen(request, timeout=60) as response:
            range_honored = not resume_from or getattr(response, "status", None) == 206
            mode = "ab" if resume_from and range_honored else "wb"
            starting_size = resume_from if mode == "ab" else 0
            content_length = response.headers.get("Content-Length")
            expected_size = starting_size + int(content_length) if content_length else None
            if expected_size is not None and expected_size > max_bytes:
                if part.exists():
                    part.unlink()
                raise ValueError(f"mmcif_too_large:{pdb_id}:{expected_size}")
            written = starting_size
            with part.open(mode) as handle:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        handle.close()
                        part.unlink(missing_ok=True)
                        raise ValueError(f"mmcif_too_large:{pdb_id}:{written}")
                    handle.write(chunk)
    if not part.exists() or part.stat().st_size == 0:
        raise ValueError(f"download_failed:{pdb_id}")
    os.replace(part, path)
    return path


def extract_auth_chain_from_mmcif(
    cif_path: Path,
    auth_chain_id: str,
    max_internal_gap_ratio: float = 0.0,
) -> dict[str, Any]:
    """Extract one auth chain's sequence and coordinates from an mmCIF entry.

    ``max_internal_gap_ratio`` controls how much unmodelled interior the chain may
    have. Deposited RNA routinely leaves flexible loops unresolved, so requiring a
    gapless ``label_seq_id`` run discards otherwise usable chains: across the 149
    chains this gate rejected in the v3 train build, the observed residues aligned
    to the RNA3DB metadata sequence at 100% for every single one, with a median gap
    of only 8.8% of the span. The gaps are missing observations, not renumbering.

    At the default 0.0 the behaviour is unchanged and any gap still raises. Above
    it, gaps up to that fraction of the spanned range are tolerated; the returned
    sequence and coordinates then cover only the observed residues, which is what
    the caller already assumes, and ``internal_gap_*`` reports what was skipped so
    the omission stays visible downstream rather than silently changing geometry.
    """

    atom_index = {name: index for index, name in enumerate(RNA_ATOMS)}
    residues = _stream_auth_chain_residues(cif_path, auth_chain_id, atom_index)

    if not residues:
        raise ValueError(f"auth_chain_not_found:{auth_chain_id}")

    ordered_ids = sorted(residues)
    spanned = ordered_ids[-1] - ordered_ids[0] + 1
    internal_gap = spanned - len(ordered_ids)
    if internal_gap:
        gap_ratio = internal_gap / spanned
        if gap_ratio > max_internal_gap_ratio:
            raise ValueError(
                f"noncontiguous_label_seq_id:{internal_gap}:{gap_ratio:.3f}"
            )
    full_sequence = "".join(str(residues[seq_id]["base"]) for seq_id in ordered_ids)
    coords = torch.full((len(full_sequence), len(RNA_ATOMS), 3), float(FILL_VALUE), dtype=torch.float32)
    missing_by_index: list[list[str]] = []
    for output_index, seq_id in enumerate(ordered_ids):
        base = str(residues[seq_id]["base"])
        required = ("P", "C4'", "C1'", "N9" if base in {"A", "G"} else "N1")
        for atom_name, (_, xyz) in residues[seq_id]["atoms"].items():
            coords[output_index, atom_index[atom_name]] = torch.tensor(xyz, dtype=torch.float32)
        missing = [atom_name for atom_name in required if torch.any(coords[output_index, atom_index[atom_name]] == FILL_VALUE)]
        missing_by_index.append(missing)

    valid_indices = [index for index, missing in enumerate(missing_by_index) if not missing]
    if not valid_indices:
        raise ValueError("missing_required_atoms_all_residues")
    first_valid, last_valid = valid_indices[0], valid_indices[-1]
    internal_missing = [
        (ordered_ids[index], missing_by_index[index])
        for index in range(first_valid, last_valid + 1)
        if missing_by_index[index]
    ]
    if internal_missing:
        seq_id, missing = internal_missing[0]
        raise ValueError(f"missing_required_atoms:{seq_id}:{','.join(missing)}")
    trim_left = first_valid
    trim_right = len(ordered_ids) - last_valid - 1
    if trim_left > 2 or trim_right > 2:
        raise ValueError(f"excessive_terminal_trim:{trim_left}:{trim_right}")
    sequence = full_sequence[first_valid : last_valid + 1]
    coords = coords[first_valid : last_valid + 1].contiguous()
    trimmed_ids = ordered_ids[first_valid : last_valid + 1]
    trimmed_span = trimmed_ids[-1] - trimmed_ids[0] + 1 if trimmed_ids else 0
    trimmed_gap = trimmed_span - len(trimmed_ids)
    return {
        "sequence": sequence,
        "coords": coords,
        "label_seq_ids": trimmed_ids,
        "terminal_trim_left": trim_left,
        "terminal_trim_right": trim_right,
        "internal_gap_residues": trimmed_gap,
        "internal_gap_ratio": (trimmed_gap / trimmed_span) if trimmed_span else 0.0,
    }


def _stream_auth_chain_residues(
    cif_path: Path,
    auth_chain_id: str,
    atom_index: Mapping[str, int],
) -> dict[int, dict[str, Any]]:
    """Read only the needed mmCIF loop columns and the requested chain.

    ``MMCIF2Dict`` materializes every category and every atom in large complexes.
    RNA3DB targets are chain-level, so streaming the two relevant loop categories
    keeps both runtime and memory proportional to the selected chain.
    """

    chem_parent: dict[str, str] = {}
    residues: dict[int, dict[str, Any]] = {}
    saw_atom_site = False
    pending: str | None = None
    with cif_path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = pending if pending is not None else handle.readline()
            pending = None
            if not line:
                break
            stripped = line.strip()
            if not (stripped.startswith("_chem_comp.") or stripped.startswith("_atom_site.")):
                continue
            if len(stripped.split()) != 1:
                continue
            category = stripped.split(".", 1)[0]
            fields = [stripped.split(".", 1)[1]]
            first_data = ""
            while True:
                next_line = handle.readline()
                if not next_line:
                    break
                next_stripped = next_line.strip()
                if next_stripped.startswith(f"{category}.") and len(next_stripped.split()) == 1:
                    fields.append(next_stripped.split(".", 1)[1])
                    continue
                first_data = next_line
                break

            if category == "_atom_site":
                saw_atom_site = True
            field_index = {name: offset for offset, name in enumerate(fields)}
            row_line = first_data
            while row_line:
                row_stripped = row_line.strip()
                if row_stripped == "#":
                    break
                if row_stripped == "loop_" or row_stripped.startswith("_") or row_stripped.startswith("data_"):
                    pending = row_line
                    break
                if row_stripped:
                    if category == "_chem_comp":
                        tokens = _cif_row_tokens(row_stripped, len(fields))
                        if tokens is not None:
                            _record_chem_parent(fields, tokens, chem_parent)
                    elif category == "_atom_site":
                        tokens = _cif_row_tokens(row_stripped, len(fields))
                        if tokens is not None:
                            _record_chain_atom(field_index, tokens, auth_chain_id, chem_parent, atom_index, residues)
                row_line = handle.readline()

    if not saw_atom_site:
        raise ValueError("mmcif_no_atom_site")
    return residues


def _cif_row_tokens(line: str, field_count: int) -> list[str] | None:
    tokens = line.split()
    if len(tokens) != field_count:
        try:
            tokens = shlex.split(line, posix=True, comments=False)
        except ValueError:
            return None
    if len(tokens) != field_count:
        return None
    return [_unquote_cif_token(token) for token in tokens]


def _unquote_cif_token(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _record_chem_parent(fields: Sequence[str], tokens: Sequence[str], parents: dict[str, str]) -> None:
    values = dict(zip(fields, tokens))
    comp_id = values.get("id", "").upper()
    parent = values.get("mon_nstd_parent_comp_id", "").upper()
    if comp_id and parent not in {"", ".", "?"}:
        parents[comp_id] = parent


def _record_chain_atom(
    field_index: Mapping[str, int],
    tokens: Sequence[str],
    auth_chain_id: str,
    chem_parent: Mapping[str, str],
    atom_index: Mapping[str, int],
    residues: dict[int, dict[str, Any]],
) -> None:
    def value(name: str, default: str = "") -> str:
        offset = field_index.get(name)
        return default if offset is None else tokens[offset]

    if value("pdbx_PDB_model_num", "1") not in {"1", ".", "?"} or value("auth_asym_id") != auth_chain_id:
        return
    label_seq_id = value("label_seq_id")
    if label_seq_id in {"", ".", "?"}:
        return
    atom_name = value("label_atom_id")
    comp_id = value("label_comp_id")
    atom_name = MODIFIED_ATOM_ALIASES.get(comp_id.upper(), {}).get(atom_name, atom_name)
    if atom_name not in atom_index:
        return
    base = normalize_residue(comp_id, chem_parent)
    if base is None:
        raise ValueError(f"noncanonical_residue:{comp_id}")
    seq_id = int(label_seq_id)
    residue = residues.setdefault(seq_id, {"base": base, "atoms": {}})
    if residue["base"] != base:
        raise ValueError(f"conflicting_residue:{label_seq_id}")
    altloc = value("label_alt_id", ".")
    xyz = (float(value("Cartn_x", "nan")), float(value("Cartn_y", "nan")), float(value("Cartn_z", "nan")))
    existing = residue["atoms"].get(atom_name)
    if existing is None or _altloc_rank(altloc) < _altloc_rank(existing[0]):
        residue["atoms"][atom_name] = (altloc, xyz)


def load_mmcif_tables(cif_path: Path) -> dict[str, list[dict[str, str]]]:
    """Load the mmCIF loop categories needed here with Biopython's STAR parser.

    Production RCSB files can contain semicolon-delimited multiline values and
    quoted tokens spanning physical lines.  ``MMCIF2Dict`` handles that grammar;
    the lightweight ``parse_mmcif_loops`` helper below is retained only for
    small synthetic fixtures and backwards-compatible callers.
    """

    payload = MMCIF2Dict(str(cif_path))
    return {
        table: _mmcif_category_rows(payload, table)
        for table in ("_chem_comp", "_atom_site")
    }


def _mmcif_category_rows(payload: Mapping[str, Any], table: str) -> list[dict[str, str]]:
    prefix = f"{table}."
    columns: dict[str, list[str]] = {}
    for key, raw_values in payload.items():
        if not str(key).startswith(prefix):
            continue
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        columns[str(key)[len(prefix) :]] = [str(value) for value in values]
    if not columns:
        return []
    row_count = max(len(values) for values in columns.values())
    rows: list[dict[str, str]] = []
    for index in range(row_count):
        row: dict[str, str] = {}
        for name, values in columns.items():
            if len(values) == row_count:
                row[name] = values[index]
            elif len(values) == 1:
                row[name] = values[0]
        rows.append(row)
    return rows


def parse_mmcif_loops(text: str) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = defaultdict(list)
    lines = iter(text.splitlines())
    for line in lines:
        if line.strip() != "loop_":
            continue
        fields: list[str] = []
        values: list[list[str]] = []
        for loop_line in lines:
            stripped = loop_line.strip()
            if not stripped:
                continue
            if stripped.startswith("_"):
                fields.append(stripped.split()[0])
                continue
            if stripped == "loop_" or stripped.startswith("data_"):
                break
            if not fields:
                continue
            values.append(shlex.split(stripped, posix=True))
        if not fields:
            continue
        table_name = fields[0].split(".", 1)[0]
        column_names = [field.split(".", 1)[1] for field in fields]
        for row_values in values:
            if len(row_values) < len(column_names):
                continue
            tables[table_name].append(dict(zip(column_names, row_values[: len(column_names)])))
    return dict(tables)


def write_single_chain_pdb(path: Path, sequence: str, coords: torch.Tensor) -> None:
    atom_index = {name: index for index, name in enumerate(RNA_ATOMS)}
    serial = 1
    lines: list[str] = []
    for residue_index, base in enumerate(sequence, start=1):
        for atom_name in RNA_ATOMS:
            xyz = coords[residue_index - 1, atom_index[atom_name]]
            if not torch.isfinite(xyz).all() or torch.any(torch.isclose(xyz, torch.tensor(float(FILL_VALUE)))):
                continue
            x, y, z = (float(value) for value in xyz)
            lines.append(
                f"ATOM  {serial:5d} {atom_name:>4s} {base:>3s} A"
                f"{residue_index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}          {atom_name[0]:>2s}"
            )
            serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class NearHomologyIndex:
    def __init__(self, sequences: Iterable[str], *, threshold: float = 0.80, k: int = 6) -> None:
        self.threshold = threshold
        self.k = k
        self.sequences = sorted({normalize_sequence(sequence) for sequence in sequences if normalize_sequence(sequence)}, key=len)
        self.by_kmer: dict[str, set[int]] = defaultdict(set)
        for index, sequence in enumerate(self.sequences):
            for kmer in kmers(sequence, self.k):
                self.by_kmer[kmer].add(index)

    def find_near_match(self, sequence: str) -> dict[str, Any] | None:
        query = normalize_sequence(sequence)
        if not query:
            return None
        candidate_indices = self._candidate_indices(query)
        best: dict[str, Any] | None = None
        for index in candidate_indices:
            other = self.sequences[index]
            longer = max(len(query), len(other))
            if longer == 0:
                continue
            similarity = lcs_length(query, other) / longer
            if best is None or similarity > best["similarity"]:
                best = {"similarity": similarity, "sequence_hash": sequence_hash(other)}
            if similarity >= self.threshold:
                return best
        return None

    def _candidate_indices(self, sequence: str) -> list[int]:
        # A high LCS does not mathematically guarantee a shared contiguous
        # k-mer.  The Biopython scorer used by ``lcs_length`` is fast enough to
        # screen the full reference set, so do not trade leakage safety for a
        # heuristic prefilter here.
        return list(range(len(self.sequences)))


def lcs_length(left: str, right: str) -> int:
    return int(round(_LCS_ALIGNER.score(left, right)))


def kmers(sequence: str, k: int) -> set[str]:
    if k <= 0 or len(sequence) < k:
        return set()
    return {sequence[index : index + k] for index in range(len(sequence) - k + 1)}


def load_processed_sequences(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return collect_sequences(payload)


def collect_sequences(payload: Any) -> set[str]:
    sequences: set[str] = set()
    if isinstance(payload, Mapping):
        if "sequence" in payload:
            normalized = normalize_sequence(str(payload["sequence"]))
            if normalized:
                sequences.add(normalized)
        for value in payload.values():
            sequences.update(collect_sequences(value))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            sequences.update(collect_sequences(value))
    return sequences


def load_ribodiffusion_exposed_chains(paths: Sequence[str | Path]) -> set[str]:
    exposed: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            continue
        if path.suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            exposed.update(collect_pdb_chains_from_text(path.read_text(encoding="utf-8")))
            continue
        exposed.update(collect_pdb_chains(payload))
    return exposed


def collect_pdb_chains_from_text(text: str) -> set[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return set()
    header = re.split(r"\s+", lines[0])
    header_upper = [value.upper() for value in header]
    if "PDB_ID" in header_upper:
        pdb_index = header_upper.index("PDB_ID")
        return {
            normalized
            for line in lines[1:]
            if len((parts := re.split(r"\s+", line))) > pdb_index
            if (normalized := normalize_pdb_chain_text(parts[pdb_index]))
        }
    return {
        normalized
        for line in lines
        if (normalized := normalize_pdb_chain_text(line))
        if re.fullmatch(r"[0-9A-Z]{4}:.+", normalized)
    }


def collect_pdb_chains(payload: Any) -> set[str]:
    chains: set[str] = set()
    if isinstance(payload, Mapping):
        pdb = _value_for_any_key(payload, ("pdb_id", "pdb", "pdbid", "entry_id"))
        chain = _value_for_any_key(payload, ("chain_id", "auth_chain_id", "chain", "auth_asym_id"))
        if pdb is not None and chain is not None:
            chains.add(normalize_pdb_chain(str(pdb), str(chain)))
        for value in payload.values():
            chains.update(collect_pdb_chains(value))
    elif isinstance(payload, str):
        normalized = normalize_pdb_chain_text(payload)
        if normalized:
            chains.add(normalized)
    elif isinstance(payload, Sequence):
        for value in payload:
            chains.update(collect_pdb_chains(value))
    return chains


def normalize_sequence(sequence: str) -> str:
    return sequence.upper().replace("T", "U").replace(" ", "")


def normalize_pdb_chain(pdb_id: str, chain_id: str) -> str:
    return f"{pdb_id.upper()}:{chain_id}"


def normalize_pdb_chain_text(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if ":" in raw:
        pdb, chain = raw.split(":", 1)
        return normalize_pdb_chain(pdb, chain)
    if "_" in raw:
        pdb, chain = raw.split("_", 1)
        return normalize_pdb_chain(pdb, chain)
    if "-" in raw:
        pdb, chain = raw.split("-", 1)
        return normalize_pdb_chain(pdb, chain)
    return raw.upper()


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(normalize_sequence(sequence).encode("utf-8")).hexdigest()


def stable_rna3db_target_id(candidate: RNA3DBCandidate, sequence: str, digest: str) -> str:
    short = hashlib.sha256(f"{candidate.pdb_chain}:{sequence}:{digest}".encode("utf-8")).hexdigest()[:10]
    return f"rna3db_{candidate.pdb_id.lower()}_{_safe_id(candidate.auth_chain_id)}_{short}"


def normalize_residue(comp_id: str, chem_parent: Mapping[str, str]) -> str | None:
    comp = comp_id.upper()
    if comp in CANONICAL_BASES:
        return comp
    parent = chem_parent.get(comp, "").upper()
    if parent in CANONICAL_BASES:
        return parent
    return COMMON_MODIFIED_BASE_PARENTS.get(comp)


def _candidate_from_chain(
    chain: Mapping[str, Any],
    component: str,
    cluster_id: str,
    chain_hint: str | None = None,
) -> RNA3DBCandidate | None:
    pdb_id = _value_for_any_key(chain, ("pdb_id", "pdb", "pdbid", "entry_id"))
    chain_id = _value_for_any_key(chain, ("auth_chain_id", "chain_id", "chain", "auth_asym_id"))
    if pdb_id is None or chain_id is None:
        pdb_chain = _value_for_any_key(chain, ("pdb_chain", "pdb_chain_id", "id")) or chain_hint
        if isinstance(pdb_chain, str) and ("_" in pdb_chain or ":" in pdb_chain or "-" in pdb_chain):
            normalized = normalize_pdb_chain_text(pdb_chain)
            pdb_id, chain_id = normalized.split(":", 1)
    sequence = normalize_sequence(str(_value_for_any_key(chain, ("sequence", "seq", "rna_sequence")) or ""))
    release_date = str(_value_for_any_key(chain, ("release_date", "date", "pdb_release_date")) or "")
    resolution = _parse_float(_value_for_any_key(chain, ("resolution", "resolution_angstrom", "pdb_resolution")))
    if pdb_id is None or chain_id is None or resolution is None:
        return None
    return RNA3DBCandidate(
        pdb_id=str(pdb_id).upper(),
        auth_chain_id=str(chain_id),
        component=component,
        cluster_id=cluster_id,
        sequence=sequence,
        release_date=release_date,
        resolution=resolution,
        metadata=dict(chain),
    )


def _metadata_rejection(candidate: RNA3DBCandidate, config: BuildRNA3DBConfig, cutoff: date) -> str | None:
    if not candidate.sequence or any(base not in CANONICAL_BASES for base in candidate.sequence):
        return "noncanonical_metadata_sequence"
    if len(candidate.sequence) < config.min_length:
        return "too_short"
    if len(candidate.sequence) > config.max_length:
        return "too_long"
    if candidate.resolution > config.max_resolution:
        return "resolution_too_low"
    try:
        released = date.fromisoformat(candidate.release_date[:10])
    except ValueError:
        return "release_date_missing"
    if released < cutoff:
        return "release_date_cutoff"
    return None


def _reject(counter: Counter[str], records: list[dict[str, Any]], candidate: RNA3DBCandidate, reason: str, **extra: Any) -> None:
    counter[reason] += 1
    records.append(
        {
            "reason": reason,
            "pdb_id": candidate.pdb_id,
            "auth_chain_id": candidate.auth_chain_id,
            "pdb_chain": candidate.pdb_chain,
            "component": candidate.component,
            "cluster_id": candidate.cluster_id,
            "sequence_hash": sequence_hash(candidate.sequence),
            **extra,
        }
    )


def _reject_cluster(counter: Counter[str], records: list[dict[str, Any]], candidates: Sequence[RNA3DBCandidate], reason: str) -> None:
    for candidate in candidates:
        _reject(counter, records, candidate, reason)


def _cluster_items(component: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    for key, value in component.items():
        lowered = str(key).lower()
        if "99" in lowered and "cluster" in lowered:
            return [(name, item) for name, item in _named_items(value) if isinstance(item, Mapping)]
    clusters = _value_for_any_key(component, ("clusters", "cluster"))
    if isinstance(clusters, Mapping):
        for key in ("99%", "99", "99% cluster", "cluster_99", "clusters_99"):
            if key in clusters:
                return [(name, item) for name, item in _named_items(clusters[key]) if isinstance(item, Mapping)]
    metadata_keys = {"component", "component_id", "name", "id", "description"}
    direct_clusters: list[tuple[str, Mapping[str, Any]]] = []
    for name, item in _named_items(component):
        if name.lower() in metadata_keys or not isinstance(item, Mapping):
            continue
        if _candidate_from_chain(item, "", "") is None:
            direct_clusters.append((name, item))
    return direct_clusters


def _named_items(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        return [(str(key), item) for key, item in value.items()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [(str(index), item) for index, item in enumerate(value)]
    return [("0", value)]


def _value_for_any_key(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _chem_comp_parents(tables: Mapping[str, list[dict[str, str]]]) -> dict[str, str]:
    parents: dict[str, str] = {}
    for row in tables.get("_chem_comp", []):
        comp_id = _row_value(row, "id", "").upper()
        parent = _row_value(row, "mon_nstd_parent_comp_id", "").upper()
        if comp_id and parent not in {"", ".", "?"}:
            parents[comp_id] = parent
    return parents


def _row_value(row: Mapping[str, str], key: str, default: str) -> str:
    value = str(row.get(key, default))
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _altloc_rank(value: str) -> tuple[int, str]:
    if value in {"", ".", "?"}:
        return (0, value)
    if value == "A":
        return (1, value)
    return (2, value)


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)
