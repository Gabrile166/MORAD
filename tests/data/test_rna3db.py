from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.constants import RNA_ATOMS
from src.data.rna3db import (
    BuildRNA3DBConfig,
    NearHomologyIndex,
    collect_pdb_chains,
    collect_pdb_chains_from_text,
    download_mmcif,
    COMMON_MODIFIED_BASE_PARENTS,
    extract_auth_chain_from_mmcif,
    index_local_chain_mmcifs,
    load_mmcif_tables,
    lcs_length,
    normalize_residue,
    parse_train_candidates,
    select_rna3db_targets,
    write_rna3db_outputs,
)


def test_parse_train_candidates_reads_component_cluster_and_chain_metadata() -> None:
    split = {
        "train_set": {
            "component": [
                {
                    "component_id": "ribosome",
                    "99% cluster": [
                        {
                            "cluster_id": "cluster-1",
                            "chains": [
                                _chain("1ABC", "B", "AUGC" * 8, resolution=2.5),
                            ],
                        }
                    ],
                }
            ]
        }
    }

    candidates = parse_train_candidates(split)

    assert len(candidates) == 1
    assert candidates[0].component == "ribosome"
    assert candidates[0].cluster_id == "cluster-1"
    assert candidates[0].pdb_chain == "1ABC:B"


def test_parse_train_candidates_accepts_official_direct_component_mapping() -> None:
    split = {
        "train_set": {
            "component_1": {
                "cluster-representative": {
                    "chain_A": _chain("4XYZ", "A", "AUGC" * 8, resolution=2.0),
                    "chain_B": _chain("4XYZ", "B", "UGCA" * 8, resolution=2.2),
                }
            }
        }
    }

    candidates = parse_train_candidates(split)

    assert {candidate.cluster_id for candidate in candidates} == {"cluster-representative"}
    assert {candidate.pdb_chain for candidate in candidates} == {"4XYZ:A", "4XYZ:B"}


@pytest.mark.parametrize("modified,parent", [("1MG", "G"), ("2MU", "U"), ("H2U", "U"), ("OMG", "G"), ("OMU", "U")])
def test_common_modified_residues_have_canonical_parents(modified: str, parent: str) -> None:
    assert normalize_residue(modified, {}) == parent


def test_tsv_header_extracts_only_pdb_id_column() -> None:
    text = "SET PDB_ID CLUSTER_ID\ntrain 1ABC_A cluster-1\ntest 2BCD:B cluster-2\n"

    assert collect_pdb_chains_from_text(text) == {"1ABC:A", "2BCD:B"}


def test_extract_auth_chain_normalizes_modified_residue_and_requires_atoms(tmp_path: Path) -> None:
    cif = tmp_path / "demo.cif"
    cif.write_text(_mmcif("A", "m1A", parent="A"), encoding="utf-8")

    extracted = extract_auth_chain_from_mmcif(cif, "A")

    assert extracted["sequence"] == "A"
    coords = extracted["coords"]
    atom_index = {name: index for index, name in enumerate(RNA_ATOMS)}
    assert coords.shape == (1, len(RNA_ATOMS), 3)
    assert torch.isfinite(coords[0, atom_index["P"]]).all()
    assert torch.isfinite(coords[0, atom_index["N9"]]).all()


def test_extract_auth_chain_rejects_missing_required_atom(tmp_path: Path) -> None:
    cif = tmp_path / "missing.cif"
    cif.write_text(_mmcif("A", "A", omit=("C1'",)), encoding="utf-8")

    with pytest.raises(ValueError, match="missing_required_atoms"):
        extract_auth_chain_from_mmcif(cif, "A")


def test_extract_auth_chain_trims_terminal_residue_with_missing_phosphate(tmp_path: Path) -> None:
    cif = tmp_path / "terminal-missing.cif"
    lines = _mmcif_chain("A", "AU").splitlines()
    cif.write_text("\n".join(line for line in lines if not line.startswith("ATOM 1 ")) + "\n", encoding="utf-8")

    extracted = extract_auth_chain_from_mmcif(cif, "A")

    assert extracted["sequence"] == "U"
    assert extracted["terminal_trim_left"] == 1
    assert extracted["terminal_trim_right"] == 0


def test_extract_auth_chain_rejects_interior_gap_by_default(tmp_path: Path) -> None:
    """A gapless run stays the default: strictness is opt-out, not opt-in."""

    cif = tmp_path / "gap-default.cif"
    lines = _mmcif_chain("A", "AUGC").splitlines()
    # Drop every atom of residue 2, leaving label_seq_id 1,3,4.
    kept = [line for line in lines if not (line.startswith("ATOM") and line.split()[8] == "2")]
    cif.write_text("\n".join(kept) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="noncontiguous_label_seq_id"):
        extract_auth_chain_from_mmcif(cif, "A")


def test_extract_auth_chain_tolerates_interior_gap_within_ratio(tmp_path: Path) -> None:
    """Above the threshold the chain survives and reports the skipped residue.

    The returned sequence covers observed residues only, so a caller that trusts
    ``sequence``/``coords`` to be positionally aligned still gets a consistent pair.
    """

    cif = tmp_path / "gap-allowed.cif"
    lines = _mmcif_chain("A", "AUGC").splitlines()
    kept = [line for line in lines if not (line.startswith("ATOM") and line.split()[8] == "2")]
    cif.write_text("\n".join(kept) + "\n", encoding="utf-8")

    extracted = extract_auth_chain_from_mmcif(cif, "A", max_internal_gap_ratio=0.30)

    assert extracted["sequence"] == "AGC"
    assert extracted["coords"].shape[0] == len("AGC")
    assert extracted["label_seq_ids"] == [1, 3, 4]
    assert extracted["internal_gap_residues"] == 1
    assert extracted["internal_gap_ratio"] == pytest.approx(0.25)


def test_extract_auth_chain_still_rejects_gap_above_ratio(tmp_path: Path) -> None:
    """The threshold is a real bound, not a blanket bypass."""

    cif = tmp_path / "gap-too-big.cif"
    lines = _mmcif_chain("A", "AUGC").splitlines()
    kept = [
        line
        for line in lines
        if not (line.startswith("ATOM") and line.split()[8] in {"2", "3"})
    ]
    cif.write_text("\n".join(kept) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="noncontiguous_label_seq_id"):
        extract_auth_chain_from_mmcif(cif, "A", max_internal_gap_ratio=0.30)


def test_gapless_chain_reports_zero_gap(tmp_path: Path) -> None:
    """Relaxing the gate must not perturb chains that never had a gap."""

    cif = tmp_path / "contiguous.cif"
    cif.write_text(_mmcif_chain("A", "AUGC"), encoding="utf-8")

    strict = extract_auth_chain_from_mmcif(cif, "A")
    relaxed = extract_auth_chain_from_mmcif(cif, "A", max_internal_gap_ratio=0.30)

    assert strict["sequence"] == relaxed["sequence"] == "AUGC"
    assert torch.equal(strict["coords"], relaxed["coords"])
    assert relaxed["internal_gap_residues"] == 0


def test_modified_base_parents_cover_rna_linking_codes_only() -> None:
    """Parent map is sourced from the RCSB CCD, so DNA codes must stay excluded.

    A DNA-linking residue inside an RNA chain means a hybrid, not a modified RNA
    base; mapping it to a canonical base would silently admit the wrong molecule.
    """

    assert COMMON_MODIFIED_BASE_PARENTS["PSU"] == "U"
    assert COMMON_MODIFIED_BASE_PARENTS["1MA"] == "A"
    assert COMMON_MODIFIED_BASE_PARENTS["7MG"] == "G"
    assert COMMON_MODIFIED_BASE_PARENTS["CCC"] == "C"
    for dna_code in ("DA", "DC", "DG", "DU", "CFL", "CSL", "AF2", "G47", "UFT"):
        assert dna_code not in COMMON_MODIFIED_BASE_PARENTS
    assert set(COMMON_MODIFIED_BASE_PARENTS.values()) <= {"A", "C", "G", "U"}


def test_standard_mmcif_parser_accepts_multiline_values(tmp_path: Path) -> None:
    cif = tmp_path / "multiline.cif"
    atom_cif = _mmcif("A", "A")
    cif.write_text(
        atom_cif.replace(
            "data_demo\n",
            "data_demo\n_struct.title\n;A title\nthat spans lines\n;\n",
            1,
        ),
        encoding="utf-8",
    )

    tables = load_mmcif_tables(cif)
    extracted = extract_auth_chain_from_mmcif(cif, "A")

    assert len(tables["_atom_site"]) == 4
    assert extracted["sequence"] == "A"


def test_near_homology_uses_lcs_over_longer_length() -> None:
    assert lcs_length("AUGCAUGC", "AUGGCAUC") == 7
    index = NearHomologyIndex(["GGAUGCAUGC"], threshold=0.80, k=4)

    match = index.find_near_match("AUGCAUGC")

    assert match is not None
    assert match["similarity"] >= 0.80


def test_near_homology_does_not_reject_long_target_for_short_subsequence() -> None:
    index = NearHomologyIndex(["UUUUAAUUUUC"], threshold=0.80, k=4)

    assert index.find_near_match("GGGUUUUAAUUUUCCCCCCCCCCCCCCCCCCCC") is None


def test_download_mmcif_reuses_cache_and_passes_resume_offset(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    def fetch(pdb_id: str, destination: Path, resume_from: int) -> None:
        calls.append((pdb_id, resume_from))
        mode = "ab" if resume_from else "wb"
        with destination.open(mode) as handle:
            handle.write(b"data_demo\n")

    first = download_mmcif("1abc", tmp_path, fetch_mmcif=fetch)
    second = download_mmcif("1abc", tmp_path, fetch_mmcif=fetch)

    assert first == second
    assert calls == [("1ABC", 0)]
    assert first.read_text(encoding="utf-8") == "data_demo\n"


def test_download_mmcif_rejects_and_removes_oversized_partial(tmp_path: Path) -> None:
    def fetch(_pdb_id: str, destination: Path, _resume_from: int) -> None:
        destination.write_bytes(b"x" * 11)

    with pytest.raises(ValueError, match="mmcif_too_large"):
        download_mmcif("1abc", tmp_path, fetch_mmcif=fetch, max_bytes=10)

    assert not (tmp_path / "1abc.cif.part").exists()


def test_index_local_chain_mmcifs_reads_official_hierarchy(tmp_path: Path) -> None:
    path = tmp_path / "train_set" / "component_1" / "cluster_1" / "1abc_A.cif"
    path.parent.mkdir(parents=True)
    path.write_text("data_demo\n", encoding="utf-8")

    index = index_local_chain_mmcifs(tmp_path)

    assert index == {"1ABC:A": path}


def test_select_rna3db_targets_filters_exports_and_writes_manifest(tmp_path: Path) -> None:
    selected_sequence = "AUGC" * 8
    good_sequence = "CCCCCCCCUUUUUUUUGGGGGGGGAAAAAAAA"
    near_sequence = "G" + selected_sequence[:-1]
    split = {
        "train_set": {
            "components": [
                {
                    "id": "comp",
                    "clusters": {
                        "99%": [
                            {
                                "id": "exposed-cluster",
                                "chains": [_chain("9ZZZ", "X", selected_sequence, resolution=1.0)],
                            },
                                {
                                    "id": "cluster-good",
                                    "chains": [
                                        _chain("1AAA", "A", selected_sequence, resolution=3.0),
                                        _chain("1AAB", "A", good_sequence, resolution=2.0),
                                    ],
                                },
                            {
                                "id": "cluster-near",
                                "chains": [_chain("2AAA", "A", near_sequence, resolution=2.0)],
                            },
                            {
                                "id": "cluster-exact",
                                "chains": [_chain("3AAA", "A", "CUGA" * 8, resolution=2.0)],
                            },
                            {
                                "id": "cluster-old",
                                "chains": [_chain("4AAA", "A", "GUCA" * 8, resolution=2.0, release_date="2024-12-31")],
                            },
                        ]
                    },
                }
            ]
        }
    }
    cif_by_pdb = {
        "1AAB": _mmcif_chain("A", good_sequence),
        "2AAA": _mmcif_chain("A", near_sequence),
    }

    def fetch(pdb_id: str, destination: Path, resume_from: int) -> None:
        destination.write_text(cif_by_pdb[pdb_id], encoding="utf-8")

    result = select_rna3db_targets(
        split,
        config=BuildRNA3DBConfig(limit=1, overfetch=3, homology_threshold=0.80, homology_kmer=4),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        processed_sequences=["CUGA" * 8, selected_sequence],
        ribodiffusion_exposed_chains=collect_pdb_chains({"heldout": [{"pdb_id": "9ZZZ", "chain_id": "X"}]}),
        fetch_mmcif=fetch,
    )

    assert result["summary"]["selected"] == 1
    assert result["summary"]["rejected"]["ribodiffusion_cluster_exposed"] == 1
    assert result["summary"]["rejected"]["processed_exact_sequence"] == 2
    assert result["summary"]["rejected"]["processed_near_homology"] == 1
    assert result["summary"]["rejected"]["release_date_cutoff"] == 1
    target = result["targets"][0]
    target_id = target["metadata"]["target_id"]
    pdb_path = tmp_path / "out" / "pdbs" / f"{target_id}.pdb"
    assert pdb_path.exists()
    assert pdb_path.name == f"{target_id}.pdb"
    assert " A   1" in pdb_path.read_text(encoding="utf-8")
    assert target["raw_record"]["_rna3db"]["pdb_path"] == str(pdb_path)
    assert target["raw_record"]["id_list"] == [target_id]
    assert target["raw_record"]["coords_list"][0].shape == (32, len(RNA_ATOMS), 3)

    paths = write_rna3db_outputs(result, tmp_path / "out")
    loaded = torch.load(paths["manifest"], map_location="cpu", weights_only=False)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "ride_rl_target_pool.v1"
    assert loaded["targets"][0]["metadata"]["source_split"] == "rna3db_train"
    assert selection[0]["pdb_path"] == str(pdb_path)


def test_select_rna3db_targets_accepts_small_terminal_metadata_trim(tmp_path: Path) -> None:
    sequence = "AUGC" * 9
    split = {
        "train_set": {
            "component_1": {
                "cluster_1": {"entry_1": _chain("1ABC", "A", sequence, resolution=2.0)}
            }
        }
    }
    lines = _mmcif_chain("A", sequence).splitlines()
    trimmed_cif = "\n".join(line for line in lines if not line.startswith("ATOM 1 ")) + "\n"

    def fetch(_pdb_id: str, destination: Path, _resume_from: int) -> None:
        destination.write_text(trimmed_cif, encoding="utf-8")

    result = select_rna3db_targets(
        split,
        config=BuildRNA3DBConfig(limit=1, max_terminal_trim=2),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        fetch_mmcif=fetch,
    )

    assert result["summary"]["selected"] == 1
    target = result["targets"][0]
    assert target["metadata"]["length"] == len(sequence) - 1
    assert target["metadata"]["rna3db"]["terminal_trim_left"] == 1


def _chain(pdb_id: str, chain: str, sequence: str, *, resolution: float, release_date: str = "2025-02-01") -> dict[str, object]:
    return {
        "pdb_id": pdb_id,
        "auth_chain_id": chain,
        "sequence": sequence,
        "release_date": release_date,
        "resolution": resolution,
    }


def _mmcif_chain(auth_chain: str, sequence: str) -> str:
    rows = []
    serial = 1
    for seq_id, base in enumerate(sequence, start=1):
        atoms = ("P", "C4'", "C1'", "N9" if base in {"A", "G"} else "N1")
        for atom in atoms:
            rows.append(
                f"ATOM {serial} {atom[0]} {_cif_quote(atom)} . {base} A 1 {seq_id} ? "
                f"{float(serial):.1f} {float(serial + 1):.1f} {float(serial + 2):.1f} "
                f"{seq_id} {base} {auth_chain} 1"
            )
            serial += 1
    return _mmcif_header() + "\n".join(rows) + "\n"


def _mmcif(auth_chain: str, residue: str, *, parent: str | None = None, omit: tuple[str, ...] = ()) -> str:
    chem = ""
    if parent is not None:
        chem = (
            "loop_\n"
            "_chem_comp.id\n"
            "_chem_comp.mon_nstd_parent_comp_id\n"
            f"{residue} {parent}\n"
        )
    atoms = ("P", "C4'", "C1'", "N9")
    rows = []
    for serial, atom in enumerate((atom for atom in atoms if atom not in omit), start=1):
        rows.append(
            f"ATOM {serial} {atom[0]} {_cif_quote(atom)} . {residue} A 1 1 ? "
            f"{float(serial):.1f} {float(serial + 1):.1f} {float(serial + 2):.1f} "
            f"1 {residue} {auth_chain} 1"
        )
    return "data_demo\n" + chem + _mmcif_header(include_data=False) + "\n".join(rows) + "\n"


def _mmcif_header(*, include_data: bool = True) -> str:
    return (
        ("data_demo\n" if include_data else "")
        +
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.pdbx_PDB_model_num\n"
    )


def _cif_quote(value: str) -> str:
    return '"' + value + '"'
