from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.adapters import InfernalCmsearchAdapter, RNAfoldAdapter
from src.evaluation.protocol import MetricStatus


def test_rnafold_adapter_parses_dot_bracket_and_energy(tmp_path) -> None:
    binary = _write_executable(
        tmp_path / "fake_rnafold.py",
        """
#!/usr/bin/env python3
import sys

if "--version" in sys.argv:
    print("RNAfold 2.6.0")
    raise SystemExit(0)

sequence = sys.stdin.read().strip()
print(sequence)
print("((..)) (-3.40)")
""",
    )

    result = RNAfoldAdapter([sys.executable, str(binary)], timeout_s=2.0).fold("AUGCAU")

    assert result.status == MetricStatus.OK
    assert result.value == pytest.approx(-3.4)
    assert result.details["dot_bracket"] == "((..))"
    assert result.details["energy"] == pytest.approx(-3.4)
    assert result.details["version"] == "RNAfold 2.6.0"


def test_rnafold_adapter_missing_binary_is_skipped() -> None:
    result = RNAfoldAdapter("/definitely/missing/RNAfold").fold("AUGC")

    assert result.status == MetricStatus.SKIPPED
    assert result.reason == "binary_missing"


def test_rnafold_adapter_accepts_pathlike_binary(tmp_path) -> None:
    binary = _write_executable(
        tmp_path / "fake_rnafold.py",
        """
#!/usr/bin/env python3
import sys
if "--version" in sys.argv:
    print("RNAfold test")
else:
    print(sys.stdin.read().strip())
    print(".... (0.00)")
""",
    )

    result = RNAfoldAdapter(binary).fold("AUGC")

    assert result.status == MetricStatus.OK


def test_rnafold_adapter_timeout_is_structured_error(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = RNAfoldAdapter("RNAfold", timeout_s=0.01).fold("AUGC")

    assert result.status == MetricStatus.ERROR
    assert result.reason == "timeout"


def test_cmsearch_adapter_skips_missing_inputs(tmp_path) -> None:
    missing_db = tmp_path / "missing.cm"

    assert InfernalCmsearchAdapter("/missing/cmsearch", database_path=missing_db, family_accession="RF00001").search("AUGC").status == MetricStatus.SKIPPED
    assert InfernalCmsearchAdapter(sys.executable, database_path=None, family_accession="RF00001").search("AUGC").reason == "database_missing"
    db = tmp_path / "rfam.cm"
    db.write_text("fake db", encoding="utf-8")
    assert InfernalCmsearchAdapter(sys.executable, database_path=db, family_accession=None).search("AUGC").reason == "family_id_missing"
    assert InfernalCmsearchAdapter("/missing/cmsearch", database_path=db, family_accession="RF00001").search("AUGC").reason == "binary_missing"


def test_cmsearch_adapter_uses_cut_ga_and_parses_family_hit(tmp_path) -> None:
    calls_path = tmp_path / "calls.txt"
    binary = _write_executable(
        tmp_path / "fake_cmsearch.py",
        f"""
from pathlib import Path
import sys

Path({str(calls_path)!r}).write_text(" ".join(sys.argv[1:]), encoding="utf-8")

if "--version" in sys.argv:
    print("cmsearch :: INFERNAL 1.1.5")
    raise SystemExit(0)

tblout = Path(sys.argv[sys.argv.index("--tblout") + 1])
tblout.write_text(
    "# tblout\\n"
    "seq1 - RF00001 RF00001 cm 1 10 1 10 + no 1 0.50 0.0 42.1 1e-08 ! target family\\n",
    encoding="utf-8",
)
""",
    )
    database = tmp_path / "rfam.cm"
    database.write_text("fake db", encoding="utf-8")

    result = InfernalCmsearchAdapter(
        [sys.executable, str(binary)],
        database_path=database,
        family_accession="RF00001",
        timeout_s=2.0,
    ).search("AUGCAUGC")

    assert result.status == MetricStatus.OK
    assert result.value == pytest.approx(1.0)
    assert result.details["used_cut_ga"] is True
    assert result.details["matched_hits"][0]["query_accession"] == "RF00001"
    assert result.details["version"] == "cmsearch :: INFERNAL 1.1.5"
    assert "--cut_ga" in calls_path.read_text(encoding="utf-8")


def test_cmsearch_adapter_accepts_family_model_name(tmp_path) -> None:
    binary = _write_executable(
        tmp_path / "fake_cmsearch.py",
        """
from pathlib import Path
import sys

if "--version" in sys.argv:
    print("cmsearch :: INFERNAL 1.1.5")
    raise SystemExit(0)

tblout = Path(sys.argv[sys.argv.index("--tblout") + 1])
tblout.write_text("seq1 - tRNA RF00005 cm 1 10 1 10 + no 1 0.50 0.0 42.1 1e-08 ! model name\\n", encoding="utf-8")
""",
    )
    database = tmp_path / "rfam.cm"
    database.write_text("fake db", encoding="utf-8")

    result = InfernalCmsearchAdapter([sys.executable, str(binary)], database_path=database, family_id="tRNA").search("AUGC")

    assert result.status == MetricStatus.OK
    assert result.value == pytest.approx(1.0)
    assert result.details["matched_hits"][0]["query_name"] == "tRNA"


def test_cmsearch_adapter_no_matching_family_returns_zero(tmp_path) -> None:
    binary = _write_executable(
        tmp_path / "fake_cmsearch.py",
        """
from pathlib import Path
import sys

if "--version" in sys.argv:
    print("cmsearch :: INFERNAL 1.1.5")
    raise SystemExit(0)

tblout = Path(sys.argv[sys.argv.index("--tblout") + 1])
tblout.write_text("seq1 - RF99999 RF99999 cm 1 10 1 10 + no 1 0.50 0.0 42.1 1e-08 ! other family\\n", encoding="utf-8")
""",
    )
    database = tmp_path / "rfam.cm"
    database.write_text("fake db", encoding="utf-8")

    result = InfernalCmsearchAdapter([sys.executable, str(binary)], database_path=database, family_accession="RF00001").search("AUGC")

    assert result.status == MetricStatus.OK
    assert result.value == pytest.approx(0.0)
    assert result.details["matched_hits"] == []


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(body.lstrip(), encoding="utf-8")
    path.chmod(0o755)
    return path
