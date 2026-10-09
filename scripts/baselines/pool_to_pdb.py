"""Write one PDB per pool target as input for RiboDiffusion.

RiboDiffusion's inference entry point takes PDB files. Every target is written
from the same reference tensors that the evaluation scores against, so all
models are conditioned on, and compared with, identical structures.

    python scripts/baselines/pool_to_pdb.py --pool artifacts/pools/test153.pt \\
        --out-dir outputs/baselines/ribodiffusion/pdb

Only backbone atoms (P, C4', N1/N9 as stored) are emitted, which is what
RiboDiffusion's featurizer reads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Column order of ref_backbone_coords, per the pool schema.
BACKBONE_ATOMS = ("P", "C4'", "N1")


def write_pdb(path: Path, sequence: str, backbone: torch.Tensor) -> int:
    """Emit a minimal single-chain PDB; returns the residue count written.

    Every residue in the sequence gets a record, even ones the pool flags in
    ``mask_coords``. That flag marks residues whose atoms are incomplete, not
    residues that are absent: the evaluator still expects a designed base at each
    position and rejects any candidate whose length differs from the target.

    Individual atoms that were never resolved are still skipped -- the builder
    stores them as an exact zero triple, and emitting those would place a real
    atom at the origin and distort the geometry RiboDiffusion conditions on.
    """
    lines: list[str] = []
    serial = 1
    written = 0
    for index, base in enumerate(sequence):
        placed = 0
        for atom_index, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_index >= backbone.shape[1]:
                break
            x, y, z = (float(v) for v in backbone[index, atom_index])
            if x == 0.0 and y == 0.0 and z == 0.0:
                continue
            lines.append(
                f"ATOM  {serial:>5} {atom_name:<4}{base:>3} A{index + 1:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           "
                f"{atom_name[0]}"
            )
            serial += 1
            placed += 1
        if placed:
            written += 1
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest = torch.load(args.pool, map_location="cpu", weights_only=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped: list[str] = []
    for target in manifest["targets"]:
        metadata = target["metadata"]
        target_id = metadata["target_id"]
        sequence = metadata.get("sequence") or ""
        backbone = target.get("ref_backbone_coords")
        if backbone is None or not sequence:
            skipped.append(target_id)
            continue
        if backbone.shape[0] != len(sequence):
            skipped.append(f"{target_id}(len {len(sequence)} vs coords {backbone.shape[0]})")
            continue
        residues = write_pdb(out_dir / f"{target_id}.pdb", sequence, backbone)
        if residues != len(sequence):
            # A short PDB makes RiboDiffusion emit a short sequence, which the
            # evaluator rejects outright. Surface it here, where the cause is
            # visible, instead of mid-evaluation.
            skipped.append(f"{target_id}({residues}/{len(sequence)} residues placed)")
            continue
        written += 1

    print(f"[out] {written} PDB written to {out_dir}")
    if skipped:
        print(f"[skip] {len(skipped)}: {', '.join(skipped[:8])}"
              + (" ..." if len(skipped) > 8 else ""))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
