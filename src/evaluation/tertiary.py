"""C1' tertiary-structure evaluation using official US-align."""

from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path
from typing import Sequence

import torch

from .protocol import MetricResult


IMPLEMENTATION = "usalign_c1prime_tm_score_same_residue_v1"


def load_pdb_atom_coords(path: str | Path, atom_names: Sequence[str] = ("C1'", "C1*")) -> torch.Tensor:
    coords: list[tuple[float, float, float]] = []
    allowed = set(atom_names)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() not in allowed:
                continue
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not coords:
        raise ValueError(f"no requested atoms {sorted(allowed)} found in PDB: {path}")
    tensor = torch.tensor(coords, dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"non-finite coordinates in PDB: {path}")
    return tensor


def write_c1prime_pdb(coords: torch.Tensor, sequence: str, path: str | Path) -> Path:
    tensor = torch.as_tensor(coords, dtype=torch.float64).cpu()
    if tensor.shape != (len(sequence), 3):
        raise ValueError("C1' coordinates must have shape [len(sequence), 3]")
    if not torch.isfinite(tensor).all():
        raise ValueError("C1' coordinates must be finite")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    residue_names = {"A": "A", "U": "U", "G": "G", "C": "C"}
    lines: list[str] = ["REMARK target C1' trace for RiboDiffusion-style evaluation\n"]
    for index, (base, xyz) in enumerate(zip(sequence, tensor.tolist()), start=1):
        if base not in residue_names:
            raise ValueError(f"invalid RNA base: {base}")
        x, y, z = xyz
        lines.append(
            f"ATOM  {index:5d}  C1' {residue_names[base]:>3s} A{index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("TER\nEND\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


class USAlignC1Prime:
    """Run US-align on corresponding C1' residues (`-TMscore 1`)."""

    def __init__(self, binary: str | Path, timeout_s: float = 120.0) -> None:
        self.binary = Path(binary)
        self.timeout_s = float(timeout_s)

    def version(self) -> str | None:
        if not self.binary.is_file():
            return None
        try:
            completed = subprocess.run(
                [str(self.binary), "-v"],
                check=False,
                capture_output=True,
                text=True,
                timeout=min(self.timeout_s, 10.0),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        text = (completed.stdout + "\n" + completed.stderr).strip()
        for line in text.splitlines():
            stripped = line.strip(" *")
            if "US-align (Version" in stripped:
                return stripped
        return text.splitlines()[0].strip() if text else None

    def compare(self, predicted_pdb: str | Path, target_pdb: str | Path) -> MetricResult:
        if not self.binary.is_file():
            return MetricResult.skipped(
                "rhofold_c1prime_tm_score",
                f"US-align binary not found: {self.binary}",
                implementation=IMPLEMENTATION,
            )
        command = [
            str(self.binary),
            str(predicted_pdb),
            str(target_pdb),
            "-mol",
            "RNA",
            "-atom",
            " C1'",
            "-TMscore",
            "1",
            "-outfmt",
            "2",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return MetricResult.error(
                "rhofold_c1prime_tm_score",
                f"US-align timed out after {self.timeout_s:g}s",
                implementation=IMPLEMENTATION,
            )
        except OSError as exc:
            return MetricResult.error(
                "rhofold_c1prime_tm_score",
                f"US-align launch failed: {exc}",
                implementation=IMPLEMENTATION,
            )
        if completed.returncode != 0:
            return MetricResult.error(
                "rhofold_c1prime_tm_score",
                f"US-align exited {completed.returncode}: {completed.stderr.strip()[-1000:]}",
                implementation=IMPLEMENTATION,
            )
        try:
            row = next(line for line in completed.stdout.splitlines() if line and not line.startswith("#"))
            fields = row.split("\t")
            tm_target = float(fields[3])
            rmsd = float(fields[4])
            aligned_length = int(fields[10])
            target_length = int(fields[9])
            if not (math.isfinite(tm_target) and math.isfinite(rmsd)):
                raise ValueError("non-finite US-align output")
        except (StopIteration, ValueError, IndexError) as exc:
            return MetricResult.error(
                "rhofold_c1prime_tm_score",
                f"cannot parse US-align tabular output: {exc}",
                details={"stdout_tail": completed.stdout[-1000:]},
                implementation=IMPLEMENTATION,
            )
        return MetricResult.ok(
            "rhofold_c1prime_tm_score",
            tm_target,
            details={
                "rmsd": rmsd,
                "aligned_length": aligned_length,
                "target_length": target_length,
                "atom": "C1'",
                "correspondence": "same_residue_index",
                "command": command,
            },
            implementation=IMPLEMENTATION,
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
