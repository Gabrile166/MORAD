"""Collect evaluation outputs into the paper's tables (Markdown).

Each run directory is produced by scripts/evaluate_test153.sh and holds
summary.json (evaluate.py), sstt.json (run_sstt.py) and, when available,
folded_scores.json (score_folded.py).

    # Tables 1-3 and the diversity table
    python scripts/eval/collect_tables.py --root outputs/eval/test153 \\
        --runs RiboDiffusion=ribodiffusion,gRNAde=grnade,RDesign=rdesign,RIDE=ride,MORAD=morad

    # Table 4 (reward comparison)
    python scripts/eval/collect_tables.py --root outputs/eval/test153 --table4 \\
        --runs RIDE=ride,"+ Structural reward"=structural_only,"+ MORAD reward"=morad_reward

    # checkpoint comparison (evaluated with configs/eval/test153_8cand.yaml)
    python scripts/eval/collect_tables.py --root outputs/eval/checkpoints --checkpoints \\
        --runs 180=update180,230=update230,340=update340,390=update390

Column definitions
    Tables 1-3: candidate 0 of each of the 153 targets (pairing metrics over the
    targets with native base pairs).
    Diversity: all 8 candidates of each target.
    Table 4: structure columns from candidate 0; sequence, pairing and
    thermodynamic columns over all candidates.
    Checkpoints: per-target mean over the 8 candidates, then mean +/- standard
    error across targets.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

# (column, source, key, decimals)
TERTIARY = [
    ("GDT-TS", "eval", "reward_c4p_gdt_ts", 4),
    ("TM(C4')", "eval", "reward_c4p_tm_score", 4),
    ("RMSD", "eval", "reward_c4p_rmsd", 4),
    ("TM(C1')", "eval", "rhofold_c1prime_tm_score", 4),
    ("Success", "eval", "reward_good", 4),
    ("Recovery", "eval", "sequence_recovery", 4),
    ("lDDT(C4')", "fold", "lddt", 4),
    ("pLDDT", "fold", "plddt", 4),
    ("Coarse clash", "fold", "clash_score", 4),
]
PAIRING = [
    ("Precision", "sstt1", "secondary_precision", 4),
    ("Recall", "sstt1", "secondary_recall", 4),
    ("F1", "sstt1", "secondary_f1", 4),
    ("MCC", "sstt1", "secondary_mcc", 4),
]
THERMO = [
    ("MFE", "sstt1", "mfe", 3),
    ("EFE", "sstt1", "ensemble_free_energy", 3),
    ("NED", "sstt1", "ensemble_defect_per_nt", 4),
    ("PE", "sstt1", "positional_entropy", 4),
    ("MFEfreq", "sstt1", "mfe_frequency", 4),
]
DIVERSITY = [
    ("Hamming", "sstt_all", "hamming_diversity", 4),
    ("3-mer", "sstt_all", "kmer_diversity_3", 4),
]
TABLE4 = [
    ("GDT", "eval", "reward_c4p_gdt_ts", 4),
    ("TM", "eval", "reward_c4p_tm_score", 4),
    ("RMSD", "eval", "reward_c4p_rmsd", 3),
    ("C1'TM", "eval", "rhofold_c1prime_tm_score", 4),
    ("MCC", "sstt_all", "secondary_mcc", 4),
    ("F1", "sstt_all", "secondary_f1", 4),
    ("ED", "sstt_all", "ensemble_defect", 2),
    ("ED/nt", "sstt_all", "ensemble_defect_per_nt", 4),
    ("Recovery", "sstt_all", "sequence_recovery", 4),
    ("H_pair", "sstt_all", "positional_entropy", 4),
    ("p_MFE", "sstt_all", "mfe_frequency", 4),
    ("MFE", "sstt_all", "mfe", 2),
    ("GC", "sstt_all", "gc_content", 4),
    ("Hamming", "sstt_all", "hamming_diversity", 4),
]


CHECKPOINTS = [
    ("GDT-TS", "reward_c4p_gdt_ts", 4),
    ("TM(C4')", "reward_c4p_tm_score", 4),
    ("RMSD", "reward_c4p_rmsd", 3),
    ("TM(C1')", "rhofold_c1prime_tm_score", 4),
    ("Recovery", "sequence_recovery", 4),
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_run(root: Path, folder: str) -> dict[str, dict[str, Any]]:
    run_dir = root / folder
    summary = _load(run_dir / "summary.json") or {}
    sstt = _load(run_dir / "sstt.json") or {}
    folded = _load(run_dir / "folded_scores.json") or {}
    sstt_run = next(iter((sstt.get("runs") or {}).values()), {})
    fold_run = (folded.get("summary") or {}).get(folder, {})
    return {
        "eval": summary.get("paper_single_sample_candidate_metrics") or {},
        "sstt1": sstt_run.get("paper_single_sample") or {},
        "sstt_all": sstt_run.get("all_candidates") or {},
        "fold": fold_run.get("candidate0") or {},
    }


def cell(run: dict[str, dict[str, Any]], source: str, key: str, decimals: int) -> str:
    entry = run.get(source, {}).get(key)
    value = entry.get("mean") if isinstance(entry, dict) else None
    return "-" if value is None else f"{value:.{decimals}f}"


def table(title: str, columns: list, runs: list[tuple[str, dict]]) -> str:
    lines = [f"### {title}", "", "| Model | " + " | ".join(c[0] for c in columns) + " |"]
    lines.append("|---|" + "---:|" * len(columns))
    for label, run in runs:
        lines.append(f"| {label} | " + " | ".join(cell(run, *c[1:]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def target_mean_cells(run_dir: Path) -> list[str]:
    """Per-target mean over all scored candidates, then mean +/- SE across targets."""
    payload = _load(run_dir / "evaluation.json") or {}
    cells = []
    for _, key, decimals in CHECKPOINTS:
        per_target: dict[str, list[float]] = defaultdict(list)
        for candidate in payload.get("candidates", []):
            entry = (candidate.get("metrics") or {}).get(key) or {}
            value = entry.get("value")
            if entry.get("status") == "ok" and isinstance(value, (int, float)) and math.isfinite(value):
                per_target[str(candidate["target_id"])].append(float(value))
        means = [sum(v) / len(v) for v in per_target.values() if v]
        if len(means) < 2:
            cells.append("-")
            continue
        mean = sum(means) / len(means)
        se = math.sqrt(sum((m - mean) ** 2 for m in means) / (len(means) - 1) / len(means))
        cells.append(f"{mean:.{decimals}f} ± {se:.{decimals}f}")
    return cells


def checkpoint_table(root: Path, runs: list[tuple[str, str]]) -> str:
    lines = [
        "### Checkpoint comparison (8 candidates per target, mean ± SE)",
        "",
        "| Update | " + " | ".join(c[0] for c in CHECKPOINTS) + " |",
        "|---|" + "---:|" * len(CHECKPOINTS),
    ]
    for label, folder in runs:
        lines.append(f"| {label} | " + " | ".join(target_mean_cells(root / folder)) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default="outputs/eval/test153")
    parser.add_argument("--runs", required=True, help="comma-separated Label=folder entries")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--table4", action="store_true", help="print the reward-comparison table")
    mode.add_argument("--checkpoints", action="store_true", help="print the checkpoint comparison")
    parser.add_argument("--out", default="", help="also write the Markdown here")
    args = parser.parse_args()

    root = Path(args.root)
    specs = []
    for chunk in args.runs.split(","):
        label, _, folder = chunk.partition("=")
        specs.append((label.strip(), (folder or label).strip()))
    runs = [(label, load_run(root, folder)) for label, folder in specs]

    if args.checkpoints:
        blocks = [checkpoint_table(root, specs)]
    elif args.table4:
        blocks = [table("Table 4: reward comparison", TABLE4, runs)]
    else:
        blocks = [
            table("Table 1: tertiary structure (candidate 0)", TERTIARY, runs),
            table("Table 2: base pairing (candidate 0)", PAIRING, runs),
            table("Table 3: thermodynamics (candidate 0)", THERMO, runs),
            table("Sequence diversity (8 candidates per target)", DIVERSITY, runs),
        ]
    text = "\n".join(blocks)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
