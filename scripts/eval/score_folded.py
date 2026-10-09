"""Score every folded candidate: C4' lDDT, pLDDT and a coarse clash score.

fold_all_candidates.py produced a RhoFold+ prediction for each unique candidate
sequence. This script compares each prediction with its target's reference C4'
trace and reports, per candidate, the global alignment metrics (GDT-TS, TM-score,
RMSD after Kabsch superposition), the superposition-free lDDT, the predictor's
mean pLDDT (0-1 scale) and a coarse C4'-level clash score (clashes per 1000
residues). The summary averages over candidate 0 of every target and over all
candidates.

Usage
-----
    python scripts/eval/score_folded.py \\
        --runs ride=outputs/eval/test153/ride,morad=outputs/eval/test153/morad \\
        --cache outputs/eval/test153/fold_cache.json \\
        --out outputs/eval/test153/folded_scores.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sstt_metrics as M  # noqa: E402
from src.rl.reward import kabsch_align, load_c4p_coords, _gdt_ts, _tm_score  # noqa: E402


def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def score_one(job: dict[str, Any]) -> dict[str, Any]:
    """Full tertiary panel for one prediction against one reference."""
    out: dict[str, Any] = {
        "candidate_id": job["candidate_id"],
        "target_id": job["target_id"],
        "index": job["index"],
        "run": job["run"],
    }

    path = job.get("structure_path")
    if not path or not Path(path).exists():
        out["error"] = "missing predicted structure"
        return out

    try:
        predicted = load_c4p_coords(Path(path))
    except Exception as exc:
        out["error"] = f"cannot read prediction: {exc}"
        return out

    predicted = np.asarray(
        predicted.detach().cpu().numpy() if torch.is_tensor(predicted) else predicted,
        dtype=float,
    )
    reference = np.asarray(job["ref_c4p"], dtype=float)
    mask = np.asarray(job["mask"], dtype=bool)

    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape[0] != reference.shape[0]:
        out["error"] = f"length mismatch pred={predicted.shape[0]} ref={reference.shape[0]}"
        return out

    valid = mask & np.isfinite(predicted).all(axis=-1) & np.isfinite(reference).all(axis=-1)
    if valid.sum() < 4:
        out["error"] = f"only {int(valid.sum())} comparable residues"
        return out

    p = predicted[valid]
    r = reference[valid]

    # the repository's helpers operate on torch tensors, so keep both views:
    # tensors for the shared alignment/scoring code, arrays for the new metrics
    p_t = torch.from_numpy(np.ascontiguousarray(p)).float()
    r_t = torch.from_numpy(np.ascontiguousarray(r)).float()

    # global metrics need a superposition
    try:
        # kabsch_align returns the mobile set in the centred frame, so the
        # reference is centred too before taking deviations.
        aligned = kabsch_align(p_t, r_t)
        r_t = r_t - r_t.mean(dim=0, keepdim=True)
        deviations = torch.linalg.norm(aligned - r_t, dim=-1)
        out["rmsd"] = float(torch.sqrt((deviations**2).mean()).item())
        out["gdt_ts"] = float(_gdt_ts(deviations))
        out["tm_score"] = float(_tm_score(deviations, int(len(r))))
    except Exception as exc:
        out["error"] = f"alignment failed: {exc}"
        return out

    # local metrics need no superposition, which is the point
    try:
        out["lddt"] = M.lddt(p, r)
    except Exception:
        out["lddt"] = None
    try:
        out["clash_score"] = M.coarse_clash_score(p)
    except Exception:
        out["clash_score"] = None

    out["plddt"] = job.get("plddt")
    out["comparable_residues"] = int(valid.sum())
    return out


def parse_runs(spec: str) -> list[tuple[str, Path]]:
    runs = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, path = chunk.partition("=")
        runs.append((name.strip(), Path(path.strip())))
    return runs


def summarise_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("gdt_ts", "tm_score", "rmsd", "lddt", "plddt", "clash_score")
    single = [r for r in rows if r["index"] == 0]
    return {
        "candidate0": {k: M.summarise([r.get(k) for r in single]) for k in keys},
        "all_candidates": {k: M.summarise([r.get(k) for r in rows]) for k in keys},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", required=True, help="comma-separated name=evaluation_dir entries")
    parser.add_argument("--cache", required=True, help="cache written by fold_all_candidates.py")
    parser.add_argument("--pool", default=str(REPO / "artifacts/pools/test153.pt"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    print(f"[cache] {len(cache)} folded sequences", flush=True)

    pool = torch.load(args.pool, map_location="cpu", weights_only=False)
    references: dict[str, dict] = {}
    for target in pool["targets"]:
        meta = target["metadata"]
        references[meta["target_id"]] = {
            "ref_c4p": target["ref_c4p_coords"].numpy(),
            "mask": target["mask_coords"].numpy(),
        }
    print(f"[pool] {len(references)} reference structures", flush=True)

    jobs: list[dict] = []
    for run, run_dir in parse_runs(args.runs):
        payload = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        for entry in payload.get("candidates", []):
            cid = entry.get("candidate_id", "")
            parts = cid.split(":")
            if len(parts) < 3:
                continue
            tid = parts[0]
            reference = references.get(tid)
            sequence = entry.get("sequence")
            if not reference or not sequence:
                continue
            record = cache.get(sequence_key(sequence))
            if not record or record.get("error"):
                continue
            try:
                index = int(parts[2])
            except ValueError:
                continue
            jobs.append(
                {
                    "candidate_id": cid,
                    "target_id": tid,
                    "index": index,
                    "run": run,
                    "structure_path": record.get("structure_path"),
                    "plddt": record.get("plddt"),
                    "ref_c4p": reference["ref_c4p"],
                    "mask": reference["mask"],
                }
            )

    print(f"[jobs] scoring {len(jobs)} candidate/reference pairs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        scored = list(executor.map(score_one, jobs, chunksize=8))

    good = [s for s in scored if not s.get("error")]
    errors = [s for s in scored if s.get("error")]
    print(f"[done] {len(good)} scored, {len(errors)} failed", flush=True)

    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in good:
        by_run.setdefault(row["run"], []).append(row)

    summary = {run: summarise_run(rows) for run, rows in by_run.items()}
    for run, block in summary.items():
        c0 = block["candidate0"]
        line = "  ".join(
            f"{label}={c0[key]['mean']:.4f}" if c0[key]["mean"] is not None else f"{label}=n/a"
            for key, label in (("lddt", "lDDT"), ("plddt", "pLDDT"), ("clash_score", "clash"))
        )
        print(f"  {run:<16} candidate 0: {line}  (n={c0['lddt']['n']})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "candidates": good}, indent=1), encoding="utf-8"
    )
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
