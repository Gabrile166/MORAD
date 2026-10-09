"""Run the SSTT metric suite over existing evaluation.json runs.

Reads the candidate sequences and per-target metadata that the MORAD
evaluator already produced, then computes every metric that does not require
re-folding in 3D: the sequence axis, the EternaFold secondary axis, the whole
thermodynamic axis, and the cross-cutting normalisations. Existing tertiary
numbers are carried through untouched so old and new columns stay comparable.

Usage
-----
    python scripts/eval/run_sstt.py --runs name=path[,name=path...] --out out.json
                                    [--workers N] [--secondary-tool eternafold|rnafold]
                                    [--candidate-limit N]

Each path is an evaluation_2d.json written by inject_native_2d.py. Per run, the
report holds an `all_candidates` block and a `paper_single_sample` block
(candidate 0 of every target).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sstt_metrics as M  # noqa: E402


# --------------------------------------------------------------------------
# reading the existing evaluation output
# --------------------------------------------------------------------------


def metric_value(container: dict | None, key: str) -> float | None:
    """Pull a numeric value out of the evaluator's {status, value} wrapper."""
    entry = (container or {}).get(key) or {}
    if entry.get("status") != "ok":
        return None
    value = entry.get("value")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def candidate_index(candidate_id: str) -> int | None:
    """Candidate ordinal lives in the third colon-separated field.

    RIDE writes 'target:seed:0000' while the precomputed RiboDiffusion adapter
    writes 'target:RiboDiffusion:00', so the field width varies.
    """
    parts = candidate_id.split(":")
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def load_run(path: str | Path) -> dict[str, Any]:
    """Index one evaluation.json by target, keeping candidates in order."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    targets: dict[str, dict[str, Any]] = {}
    for entry in payload.get("targets", []):
        tid = entry["target_id"]
        metrics = entry.get("metrics") or {}
        targets[tid] = {
            "target_id": tid,
            "length": entry.get("length"),
            "length_bucket": entry.get("length_bucket"),
            "rna_family": entry.get("rna_family") or "unknown",
            "rna_type": entry.get("rna_type"),
            "native_sequence": entry.get("native_sequence"),
            "target_secondary_structure": entry.get("target_secondary_structure"),
            "target_secondary_source": entry.get("target_secondary_source"),
            "strict_complete_condition": entry.get("strict_complete_condition"),
            "condition_imputed_residue_count": entry.get("condition_imputed_residue_count"),
            # native controls double as the oracle's ceiling on this target
            "native_gdt": metric_value(metrics, "native_reward_c4p_gdt_ts"),
            "native_tm": metric_value(metrics, "native_reward_c4p_tm_score"),
            "native_rmsd": metric_value(metrics, "native_reward_c4p_rmsd"),
            "native_c1p_tm": metric_value(metrics, "native_rhofold_c1prime_tm_score"),
            "native_secondary_f1": metric_value(metrics, "native_secondary_structure_f1"),
            "native_good": metric_value(metrics, "native_reward_good"),
            "candidates": [],
        }

    for entry in payload.get("candidates", []):
        cid = entry.get("candidate_id", "")
        tid = cid.split(":")[0]
        if tid not in targets:
            continue
        metrics = entry.get("metrics") or {}
        targets[tid]["candidates"].append(
            {
                "candidate_id": cid,
                "index": candidate_index(cid),
                "sequence": entry.get("sequence"),
                # carry existing tertiary metrics through unchanged
                "gdt_ts": metric_value(metrics, "reward_c4p_gdt_ts"),
                "tm_score": metric_value(metrics, "reward_c4p_tm_score"),
                "rmsd": metric_value(metrics, "reward_c4p_rmsd"),
                "c1p_tm": metric_value(metrics, "rhofold_c1prime_tm_score"),
                "reward_good": metric_value(metrics, "reward_good"),
                "sequence_recovery_reported": metric_value(metrics, "sequence_recovery"),
                "secondary_f1_reported": metric_value(metrics, "secondary_structure_f1"),
                "rfam_success": metric_value(metrics, "rfam_family_success"),
            }
        )

    for record in targets.values():
        record["candidates"].sort(key=lambda c: (c["index"] is None, c["index"]))

    return {
        "targets": targets,
        "generator": payload.get("generator"),
        "protocol_version": payload.get("protocol_version"),
        "source_path": str(path),
    }


# --------------------------------------------------------------------------
# per-candidate work (runs in a worker process)
# --------------------------------------------------------------------------


def score_candidate(job: dict[str, Any]) -> dict[str, Any]:
    """Compute sequence, secondary and thermodynamic metrics for one design."""
    sequence = job["sequence"]
    native = job["native_sequence"]
    target_db = job["target_secondary_structure"]
    tool = job["secondary_tool"]

    out: dict[str, Any] = {
        "candidate_id": job["candidate_id"],
        "target_id": job["target_id"],
        "index": job["index"],
    }

    if not sequence:
        out["error"] = "missing sequence"
        return out

    out["sequence_recovery"] = M.sequence_recovery(sequence, native) if native else None

    # secondary axis: forward-fold the design, compare pairings to the target
    predicted_db = None
    if tool == "eternafold":
        try:
            predicted_db = M.fold_eternafold([sequence])[0]
        except Exception as exc:
            out["secondary_error"] = str(exc)
    if predicted_db is None:
        predicted_db = M.fold_rnafold([sequence])[0]
        if tool == "eternafold":
            out["secondary_fallback"] = "rnafold"

    out["predicted_secondary"] = predicted_db
    if predicted_db and target_db:
        scores = M.secondary_agreement(predicted_db, target_db)
        out["secondary_mcc"] = scores["mcc"]
        out["secondary_f1"] = scores["f1"]
        out["secondary_precision"] = scores["precision"]
        out["secondary_recall"] = scores["recall"]
        # base-pair-level INF is the same confusion matrix, reported separately
        # because RNA-Puzzles treats interaction fidelity as its own axis
        out["inf_wc"] = scores["mcc"]

    # thermodynamic axis: one partition-function pass yields everything
    try:
        thermo = M.compute_thermo(sequence, target_db)
        out.update(thermo.as_dict())
        if thermo.notes:
            out["thermo_notes"] = thermo.notes
    except Exception as exc:
        out["thermo_error"] = str(exc)

    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def aggregate(
    run_name: str,
    run: dict[str, Any],
    scored: list[dict[str, Any]],
    k_values: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """Build per-run summaries: overall, paper-style, by tier, by family."""
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored:
        by_target[record["target_id"]].append(record)

    # tertiary metrics come from the original evaluation, keyed the same way
    tertiary: dict[str, list[dict[str, Any]]] = {
        tid: rec["candidates"] for tid, rec in run["targets"].items()
    }

    new_keys = [
        "sequence_recovery",
        "secondary_precision",
        "secondary_recall",
        "secondary_mcc",
        "secondary_f1",
        "inf_wc",
        "mfe",
        "ensemble_free_energy",
        "mfe_frequency",
        "ensemble_diversity",
        "ensemble_defect",
        "ensemble_defect_per_nt",
        "target_probability",
        "positional_entropy",
        "melting_temperature",
        "gc_content",
    ]
    old_keys = ["gdt_ts", "tm_score", "rmsd", "c1p_tm", "reward_good", "rfam_success"]

    def collect(selector) -> dict[str, Any]:
        """Aggregate over the candidates that `selector` accepts."""
        buckets: dict[str, list[float | None]] = {k: [] for k in new_keys + old_keys}
        diversity_3mer: list[float] = []
        diversity_hamming: list[float] = []
        achievement: dict[str, list[float]] = defaultdict(list)
        pass_at: dict[int, list[float]] = defaultdict(list)
        # paired design/native values, so the ratio can also be formed from the
        # two means. Mean-of-ratios blows up when a native value is near zero;
        # ratio-of-means is the robust companion figure.
        paired: dict[str, list[tuple[float, float]]] = defaultdict(list)

        for tid, records in by_target.items():
            target = run["targets"][tid]
            chosen = [r for r in records if selector(target, r)]
            if not chosen:
                continue

            for record in chosen:
                for key in new_keys:
                    buckets[key].append(record.get(key))

            # tertiary values matched by candidate index
            index_map = {c["index"]: c for c in tertiary.get(tid, [])}
            for record in chosen:
                cand = index_map.get(record["index"])
                if cand:
                    for key in old_keys:
                        buckets[key].append(cand.get(key))

            # diversity is a property of the candidate set, so it always uses
            # every candidate of the target regardless of the selector
            seqs = [r["sequence"] for r in records if r.get("sequence")]
            if len(seqs) > 1:
                d3 = M.kmer_diversity(seqs, 3)
                if d3 is not None:
                    diversity_3mer.append(d3)
                dh = M.hamming_diversity(seqs)
                if dh is not None:
                    diversity_hamming.append(dh)

            # achievement ratio against this target's own oracle ceiling
            for record in chosen:
                cand = index_map.get(record["index"])
                if not cand:
                    continue
                for metric_key, native_key in (
                    ("gdt_ts", "native_gdt"),
                    ("tm_score", "native_tm"),
                    ("c1p_tm", "native_c1p_tm"),
                ):
                    design_value = cand.get(metric_key)
                    native_value = target.get(native_key)
                    if design_value is None or native_value is None:
                        continue
                    paired[metric_key].append((design_value, native_value))
                    # Guard the per-target ratio: a native value below this
                    # floor means the oracle failed on that target, so the
                    # ratio carries no information about design quality.
                    if native_value >= 0.10:
                        ratio = M.achievement_ratio(design_value, native_value)
                        if ratio is not None:
                            achievement[metric_key].append(ratio)
                # RMSD is lower-is-better, so the ratio inverts
                design_rmsd = cand.get("rmsd")
                native_rmsd = target.get("native_rmsd")
                if design_rmsd and native_rmsd and design_rmsd > 0:
                    paired["rmsd"].append((design_rmsd, native_rmsd))
                    achievement["rmsd"].append(native_rmsd / design_rmsd)

            # pass@k over all candidates of this target (needs the full set)
            flags = []
            for cand in tertiary.get(tid, []):
                good = cand.get("reward_good")
                if good is not None:
                    flags.append(good > 0.5)
            if flags:
                n_total = len(flags)
                n_good = sum(flags)
                for k in k_values:
                    if k <= n_total:
                        value = M.pass_at_k_unbiased(n_total, n_good, k)
                        if value is not None:
                            pass_at[k].append(value)

        summary: dict[str, Any] = {
            key: M.summarise(values) for key, values in buckets.items()
        }
        summary["kmer_diversity_3"] = M.summarise(diversity_3mer)
        summary["hamming_diversity"] = M.summarise(diversity_hamming)
        for key, values in achievement.items():
            summary[f"achievement_{key}"] = M.summarise(values)
        # ratio-of-means: robust against near-zero denominators
        for key, pairs in paired.items():
            if not pairs:
                continue
            design_mean = sum(p[0] for p in pairs) / len(pairs)
            native_mean = sum(p[1] for p in pairs) / len(pairs)
            if native_mean:
                summary[f"achievement_pooled_{key}"] = {
                    "n": len(pairs),
                    "design_mean": design_mean,
                    "native_mean": native_mean,
                    "ratio": (native_mean / design_mean)
                    if key == "rmsd"
                    else (design_mean / native_mean),
                }
        for k, values in pass_at.items():
            summary[f"pass_at_{k}"] = M.summarise(values)
        return summary

    all_candidates = collect(lambda t, r: True)
    paper_single = collect(lambda t, r: r["index"] == 0)

    tiers: dict[str, Any] = {}
    for tier in ("reliable", "moderate", "weak", "unusable"):
        tiers[tier] = collect(
            lambda t, r, tier=tier: M.reliability_tier(t.get("native_gdt")) == tier
        )

    # "usable" drops the targets whose native sequence cannot be refolded
    tiers["usable_only"] = collect(
        lambda t, r: M.reliability_tier(t.get("native_gdt")) in ("reliable", "moderate")
    )

    families: dict[str, Any] = {}
    family_names = {rec["rna_family"] for rec in run["targets"].values()}
    for family in sorted(family_names):
        families[family] = collect(
            lambda t, r, family=family: t.get("rna_family") == family
        )

    buckets: dict[str, Any] = {}
    bucket_names = {
        rec.get("length_bucket") for rec in run["targets"].values() if rec.get("length_bucket")
    }
    for bucket in sorted(bucket_names):
        buckets[bucket] = collect(
            lambda t, r, bucket=bucket: t.get("length_bucket") == bucket
        )

    return {
        "run": run_name,
        "source": run["source_path"],
        "generator": run.get("generator"),
        "all_candidates": all_candidates,
        "paper_single_sample": paper_single,
        "by_reliability_tier": tiers,
        "by_rna_family": families,
        "by_length_bucket": buckets,
    }


def native_reference(run: dict[str, Any], secondary_tool: str, workers: int) -> dict[str, Any]:
    """Score the native sequences themselves to get the achievable ceiling.

    gRNAde makes the same point: folding a backbone's own native sequence
    bounds what any design can score under that predictor.
    """
    jobs = []
    for tid, record in run["targets"].items():
        if not record.get("native_sequence"):
            continue
        jobs.append(
            {
                "candidate_id": f"{tid}:native:0000",
                "target_id": tid,
                "index": -1,
                "sequence": record["native_sequence"],
                "native_sequence": record["native_sequence"],
                "target_secondary_structure": record.get("target_secondary_structure"),
                "secondary_tool": secondary_tool,
            }
        )

    scored = run_jobs(jobs, workers)
    keys = [
        "secondary_mcc",
        "secondary_f1",
        "mfe",
        "mfe_frequency",
        "ensemble_diversity",
        "ensemble_defect",
        "ensemble_defect_per_nt",
        "target_probability",
        "positional_entropy",
        "melting_temperature",
        "gc_content",
    ]
    summary = {key: M.summarise([r.get(key) for r in scored]) for key in keys}
    summary["native_gdt"] = M.summarise(
        [r.get("native_gdt") for r in run["targets"].values()]
    )
    summary["native_tm"] = M.summarise([r.get("native_tm") for r in run["targets"].values()])
    summary["native_rmsd"] = M.summarise(
        [r.get("native_rmsd") for r in run["targets"].values()]
    )
    summary["native_c1p_tm"] = M.summarise(
        [r.get("native_c1p_tm") for r in run["targets"].values()]
    )
    return {"summary": summary, "per_target": scored}


def run_jobs(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if workers <= 1:
        return [score_candidate(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(score_candidate, jobs, chunksize=4))


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        required=True,
        help="comma-separated name=path/to/evaluation.json entries",
    )
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--secondary-tool",
        choices=("eternafold", "rnafold"),
        default="eternafold",
        help="forward-folding tool for the secondary axis (RiboPO uses EternaFold)",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="only score the first N candidates per target (0 = all)",
    )
    args = parser.parse_args()

    specs = []
    for chunk in args.runs.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, path = chunk.partition("=")
        specs.append((name.strip(), path.strip()))

    report: dict[str, Any] = {
        "schema": "sstt_report.v1",
        "secondary_tool": args.secondary_tool,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metric_direction": dict(M.METRIC_DIRECTION),
        "runs": {},
    }

    native_done = False
    for name, path in specs:
        print(f"[load] {name} <- {path}", flush=True)
        run = load_run(path)
        n_targets = len(run["targets"])
        n_cands = sum(len(r["candidates"]) for r in run["targets"].values())
        print(f"       {n_targets} targets / {n_cands} candidates", flush=True)

        jobs = []
        for tid, record in run["targets"].items():
            for cand in record["candidates"]:
                if args.candidate_limit and (cand["index"] or 0) >= args.candidate_limit:
                    continue
                jobs.append(
                    {
                        "candidate_id": cand["candidate_id"],
                        "target_id": tid,
                        "index": cand["index"],
                        "sequence": cand["sequence"],
                        "native_sequence": record.get("native_sequence"),
                        "target_secondary_structure": record.get("target_secondary_structure"),
                        "secondary_tool": args.secondary_tool,
                    }
                )

        started = time.time()
        scored = run_jobs(jobs, args.workers)
        elapsed = time.time() - started
        print(
            f"       scored {len(scored)} candidates in {elapsed:.1f}s "
            f"({elapsed / max(len(scored), 1) * 1000:.1f} ms each)",
            flush=True,
        )

        # attach sequences so diversity can be computed during aggregation
        seq_lookup = {
            cand["candidate_id"]: cand["sequence"]
            for record in run["targets"].values()
            for cand in record["candidates"]
        }
        for record in scored:
            record["sequence"] = seq_lookup.get(record["candidate_id"])

        report["runs"][name] = aggregate(name, run, scored)
        report["runs"][name]["candidate_count_scored"] = len(scored)
        report["runs"][name]["seconds"] = elapsed

        if not native_done:
            print("[native] scoring native sequences for the ceiling", flush=True)
            report["native_reference"] = native_reference(
                run, args.secondary_tool, args.workers
            )["summary"]
            native_done = True

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[done] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
