from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.evaluation.protocol import MetricStatus
from src.evaluation.tertiary import USAlignC1Prime, load_pdb_atom_coords, write_c1prime_pdb


def test_write_c1prime_pdb_round_trips_c1_prime_coordinates(tmp_path) -> None:
    coords = torch.tensor([[1.0, 2.0, 3.0], [4.25, 5.5, 6.75]], dtype=torch.float64)
    path = tmp_path / "target.c1prime.pdb"

    written = write_c1prime_pdb(coords, "AU", path)

    assert written == path
    assert torch.allclose(load_pdb_atom_coords(path), coords)
    text = path.read_text(encoding="utf-8")
    assert " C1' " in text
    assert text.endswith("TER\nEND\n")


def test_write_c1prime_pdb_rejects_shape_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"shape \[len\(sequence\), 3\]"):
        write_c1prime_pdb(torch.zeros((1, 3)), "AU", tmp_path / "bad.pdb")


def test_usalign_c1prime_parses_tabular_output(tmp_path) -> None:
    binary = _write_executable(
        tmp_path / "fake_usalign.py",
        """
#!/usr/bin/env python3
import sys

if "-v" in sys.argv:
    print("USalign fake 2026")
    raise SystemExit(0)

print("# query\\ttarget\\ttm_chain1\\ttm_chain2\\trmsd\\tid1\\tid2\\tidali\\tquery_len\\ttarget_len\\taligned_len")
print("pred.pdb\\ttarget.pdb\\t0.111\\t0.8125\\t1.25\\t0\\t0\\t0\\t7\\t8\\t6")
""",
    )

    result = USAlignC1Prime(binary, timeout_s=2.0).compare(tmp_path / "pred.pdb", tmp_path / "target.pdb")

    assert result.status == MetricStatus.OK
    assert result.value == pytest.approx(0.8125)
    assert result.details["rmsd"] == pytest.approx(1.25)
    assert result.details["target_length"] == 8
    assert result.details["aligned_length"] == 6
    assert result.details["atom"] == "C1'"


def test_usalign_c1prime_missing_binary_is_skipped() -> None:
    result = USAlignC1Prime("/definitely/missing/USalign").compare("pred.pdb", "target.pdb")

    assert result.status == MetricStatus.SKIPPED
    assert "US-align binary not found" in str(result.reason)


def test_usalign_c1prime_parse_failure_is_structured_error(tmp_path) -> None:
    binary = _write_executable(
        tmp_path / "fake_usalign.py",
        """
#!/usr/bin/env python3
print("# header only")
""",
    )

    result = USAlignC1Prime(binary, timeout_s=2.0).compare(tmp_path / "pred.pdb", tmp_path / "target.pdb")

    assert result.status == MetricStatus.ERROR
    assert str(result.reason).startswith("cannot parse US-align tabular output")
    assert result.details["stdout_tail"] == "# header only\n"


def test_usalign_c1prime_nonzero_exit_is_structured_error(tmp_path) -> None:
    binary = _write_executable(
        tmp_path / "fake_usalign.py",
        """
#!/usr/bin/env python3
import sys

print("fatal alignment problem", file=sys.stderr)
raise SystemExit(42)
""",
    )

    result = USAlignC1Prime(binary, timeout_s=2.0).compare(tmp_path / "pred.pdb", tmp_path / "target.pdb")

    assert result.status == MetricStatus.ERROR
    assert "US-align exited 42" in str(result.reason)
    assert "fatal alignment problem" in str(result.reason)


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(body.lstrip(), encoding="utf-8")
    path.chmod(0o755)
    return path
