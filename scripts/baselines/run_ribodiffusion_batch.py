#!/usr/bin/env python
"""Run official RiboDiffusion inference over a resumable PDB directory.

The default path starts one worker and loads the 1 GB checkpoint only once.
``--legacy-per-target`` preserves the upstream one-process-per-PDB behavior for
debugging or compatibility checks.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--legacy-per-target", action="store_true")
    args = parser.parse_args()

    pdb_paths = sorted(args.input_dir.glob("*.pdb"))
    if not pdb_paths:
        raise ValueError(f"No PDB files found in {args.input_dir}")
    fasta_dir = args.output_dir / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    pending: list[Path] = []
    skipped = 0
    for pdb_path in pdb_paths:
        expected = [fasta_dir / f"{pdb_path.stem}_{sample}.fasta" for sample in range(args.n_samples)]
        if not args.force and all(_valid_fasta(path) for path in expected):
            skipped += 1
            continue
        pending.append(pdb_path)

    if pending and args.legacy_per_target:
        _run_legacy(args, pending)
    elif pending:
        worker = Path(__file__).with_name("ribodiffusion_batch_worker.py")
        command = [
            str(args.python),
            str(worker.resolve()),
            "--repo",
            str(args.repo.resolve()),
            "--output-dir",
            str(args.output_dir.resolve()),
            "--n-samples",
            str(args.n_samples),
        ]
        for pdb_path in pending:
            command.extend(("--input", str(pdb_path.resolve())))
        subprocess.run(command, check=True)

    for pdb_path in pending:
        expected = [fasta_dir / f"{pdb_path.stem}_{sample}.fasta" for sample in range(args.n_samples)]
        if not all(_valid_fasta(path) for path in expected):
            raise RuntimeError(f"RiboDiffusion did not write {args.n_samples} valid FASTA files for {pdb_path}")
    print(f"RiboDiffusion batch complete: generated={len(pending)}, resumed={skipped}, total={len(pdb_paths)}")


def _run_legacy(args: argparse.Namespace, pdb_paths: list[Path]) -> None:
    for index, pdb_path in enumerate(pdb_paths, start=1):
        command = [
            str(args.python),
            "main.py",
            f"--PDB_file={pdb_path.resolve()}",
            f"--save_folder={args.output_dir.resolve()}",
            f"--config.eval.n_samples={args.n_samples}",
        ]
        subprocess.run(command, cwd=args.repo, check=True)
        print(f"[{index}/{len(pdb_paths)}] completed {pdb_path.stem}", flush=True)


def _valid_fasta(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    sequence = "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    )
    return bool(sequence) and all(base in {"A", "U", "G", "C"} for base in sequence)


if __name__ == "__main__":
    main()
