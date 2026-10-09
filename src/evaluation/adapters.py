from __future__ import annotations

import re
import subprocess
import tempfile
from os import PathLike
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.protocol import MetricResult


Command = str | PathLike[str] | Sequence[str | PathLike[str]]
RNAFOLD_IMPLEMENTATION = "RNAfold"
CMSEARCH_IMPLEMENTATION = "Infernal cmsearch"


class RNAfoldAdapter:
    metric_name = "rnafold_mfe"

    def __init__(self, binary: Command = "RNAfold", timeout_s: float = 10.0, extra_args: Sequence[str] | None = None) -> None:
        self.binary = binary
        self.timeout_s = timeout_s
        self.extra_args = list(extra_args or ())

    def probe_version(self) -> str | None:
        for version_arg in ("--version", "-h"):
            try:
                completed = subprocess.run(
                    [*_command_parts(self.binary), version_arg],
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                return None
            output = "\n".join((completed.stdout, completed.stderr))
            infernal_line = next((line.strip() for line in output.splitlines() if "INFERNAL" in line), None)
            first_line = infernal_line or _first_nonempty_line(output)
            if completed.returncode == 0 and first_line:
                return first_line.removeprefix("# ")
        return None

    def fold(self, sequence: str) -> MetricResult:
        if not sequence.strip():
            return MetricResult.skipped(self.metric_name, "empty_sequence", implementation=RNAFOLD_IMPLEMENTATION)

        version = self.probe_version()
        command = [*_command_parts(self.binary), "--noPS", *self.extra_args]
        try:
            completed = subprocess.run(
                command,
                input=f"{sequence.strip()}\n",
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError:
            return MetricResult.skipped(
                self.metric_name,
                "binary_missing",
                details={"binary": _command_name(self.binary)},
                implementation=RNAFOLD_IMPLEMENTATION,
            )
        except subprocess.TimeoutExpired:
            return MetricResult.error(
                self.metric_name,
                "timeout",
                details={"timeout_s": self.timeout_s, "version": version},
                implementation=RNAFOLD_IMPLEMENTATION,
            )
        except OSError as exc:
            return MetricResult.error(
                self.metric_name,
                str(exc),
                details={"version": version},
                implementation=RNAFOLD_IMPLEMENTATION,
            )

        if completed.returncode != 0:
            return MetricResult.error(
                self.metric_name,
                "process_failed",
                details={"returncode": completed.returncode, "stderr": completed.stderr.strip(), "version": version},
                implementation=RNAFOLD_IMPLEMENTATION,
            )

        parsed = _parse_rnafold_output(completed.stdout)
        if parsed is None:
            return MetricResult.error(
                self.metric_name,
                "dot_bracket_not_found",
                details={"stdout": completed.stdout, "version": version},
                implementation=RNAFOLD_IMPLEMENTATION,
            )

        structure, energy = parsed
        if energy is None:
            return MetricResult.error(
                self.metric_name,
                "energy_not_found",
                details={"dot_bracket": structure, "version": version},
                implementation=RNAFOLD_IMPLEMENTATION,
            )

        return MetricResult.ok(
            self.metric_name,
            value=energy,
            details={"sequence": sequence.strip(), "dot_bracket": structure, "energy": energy, "version": version},
            implementation=RNAFOLD_IMPLEMENTATION,
        )


class InfernalCmsearchAdapter:
    metric_name = "infernal_cmsearch"

    def __init__(
        self,
        binary: Command = "cmsearch",
        database_path: str | Path | None = None,
        family_accession: str | None = None,
        family_id: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.binary = binary
        self.database_path = Path(database_path) if database_path is not None else None
        self.family_id = family_id or family_accession
        self.timeout_s = timeout_s

    def probe_version(self) -> str | None:
        for version_arg in ("--version", "-h"):
            try:
                completed = subprocess.run(
                    [*_command_parts(self.binary), version_arg],
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                return None
            output = "\n".join((completed.stdout, completed.stderr))
            infernal_line = next((line.strip() for line in output.splitlines() if "INFERNAL" in line), None)
            first_line = infernal_line or _first_nonempty_line(output)
            if completed.returncode == 0 and first_line:
                return first_line.removeprefix("# ")
        return None

    def search(self, sequence: str, *, sequence_id: str = "query") -> MetricResult:
        if not sequence.strip():
            return MetricResult.skipped(self.metric_name, "empty_sequence", implementation=CMSEARCH_IMPLEMENTATION)
        if self.database_path is None:
            return MetricResult.skipped(self.metric_name, "database_missing", implementation=CMSEARCH_IMPLEMENTATION)
        if not self.database_path.exists():
            return MetricResult.skipped(
                self.metric_name,
                "database_missing",
                details={"database_path": str(self.database_path)},
                implementation=CMSEARCH_IMPLEMENTATION,
            )
        if not self.family_id:
            return MetricResult.skipped(self.metric_name, "family_id_missing", implementation=CMSEARCH_IMPLEMENTATION)

        version = self.probe_version()
        with tempfile.TemporaryDirectory(prefix="rider-cmsearch-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            fasta_path = tmp_path / "query.fa"
            tblout_path = tmp_path / "cmsearch.tblout"
            fasta_path.write_text(f">{sequence_id}\n{sequence.strip()}\n", encoding="utf-8")
            command = [
                *_command_parts(self.binary),
                "--cut_ga",
                "--tblout",
                str(tblout_path),
                str(self.database_path),
                str(fasta_path),
            ]
            try:
                completed = subprocess.run(command, text=True, capture_output=True, timeout=self.timeout_s, check=False)
            except FileNotFoundError:
                return MetricResult.skipped(
                    self.metric_name,
                    "binary_missing",
                    details={"binary": _command_name(self.binary), "version": version},
                    implementation=CMSEARCH_IMPLEMENTATION,
                )
            except subprocess.TimeoutExpired:
                return MetricResult.error(
                    self.metric_name,
                    "timeout",
                    details={"timeout_s": self.timeout_s, "version": version},
                    implementation=CMSEARCH_IMPLEMENTATION,
                )
            except OSError as exc:
                return MetricResult.error(
                    self.metric_name,
                    str(exc),
                    details={"version": version},
                    implementation=CMSEARCH_IMPLEMENTATION,
                )

            if completed.returncode != 0:
                return MetricResult.error(
                    self.metric_name,
                    "process_failed",
                    details={"returncode": completed.returncode, "stderr": completed.stderr.strip(), "version": version},
                    implementation=CMSEARCH_IMPLEMENTATION,
                )

            hits = _parse_cmsearch_tblout(tblout_path)
            matching_hits = [hit for hit in hits if self.family_id in {hit.get("query_name"), hit.get("query_accession")}]
            success = any(hit.get("included") in {"!", "?", ""} for hit in matching_hits)
            return MetricResult.ok(
                self.metric_name,
                value=1.0 if success else 0.0,
                details={
                    "family_id": self.family_id,
                    "hits": hits,
                    "matched_hits": matching_hits,
                    "used_cut_ga": True,
                    "version": version,
                },
                implementation=CMSEARCH_IMPLEMENTATION,
            )


def _command_parts(command: Command) -> list[str]:
    if isinstance(command, (str, PathLike)):
        return [str(command)]
    return [str(part) for part in command]


def _command_name(command: Command) -> str:
    return _command_parts(command)[0]


def _first_nonempty_line(*chunks: str) -> str | None:
    for chunk in chunks:
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def _parse_rnafold_output(stdout: str) -> tuple[str, float | None] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        first_token = stripped.split()[0]
        if re.fullmatch(r"[().\[\]{}<>]+", first_token):
            energy_match = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*\)", stripped)
            return first_token, float(energy_match.group(1)) if energy_match else None
    return None


def _parse_cmsearch_tblout(tblout_path: Path) -> list[dict[str, Any]]:
    if not tblout_path.exists():
        return []

    hits: list[dict[str, Any]] = []
    for line in tblout_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=17)
        if len(parts) < 17:
            continue
        hits.append(
            {
                "target_name": parts[0],
                "target_accession": parts[1],
                "query_name": parts[2],
                "query_accession": parts[3],
                "score": _parse_float(parts[14]),
                "e_value": _parse_float(parts[15]),
                "included": parts[16],
                "description": parts[17] if len(parts) > 17 else "",
            }
        )
    return hits


def _parse_float(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None
