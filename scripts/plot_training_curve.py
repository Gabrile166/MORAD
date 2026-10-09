"""Plot the held-out structural scores during MORAD training (Figure 3).

Reads the background validation results of a training run
(outputs/morad/validation/async_results.jsonl) or the released curve
(results/training_curve.csv), writes the curve as CSV and plots GDT-TS,
C4' TM-score and C1' TM-score at each evaluation together with a centred
five-point moving average.

    python scripts/plot_training_curve.py --results outputs/morad/validation/async_results.jsonl
    python scripts/plot_training_curve.py --csv results/training_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# curve column -> key written by the validation worker (candidate 0 of each target)
COLUMNS = {
    "gdt": "paper_reward_c4p_gdt_ts",
    "tm": "paper_reward_c4p_tm_score",
    "rmsd": "paper_reward_c4p_rmsd",
    "c1tm": "paper_rhofold_c1prime_tm_score",
    "success": "paper_reward_good",
    "recovery": "paper_sequence_recovery",
    "diversity": "internal_diversity",
}
PLOTTED = [("gdt", "GDT-TS", "o"), ("tm", "TM-score (C4′)", "s"), ("c1tm", "TM-score (C1′)", "^")]


def from_results(path: Path) -> list[dict[str, float]]:
    latest: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[int(record["step"])] = record
    rows = []
    for step in sorted(latest):
        record = latest[step]
        row = {"step": step}
        for column, key in COLUMNS.items():
            value = record.get(key)
            row[column] = round(float(value), 4) if isinstance(value, (int, float)) else None
        rows.append(row)
    return rows


def from_csv(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [
            {k: (int(v) if k == "step" else float(v) if v not in ("", None) else None) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def centred_mean(values: list[float], window: int = 5) -> list[float]:
    half = window // 2
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - half): i + half + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path, help="async_results.jsonl of a training run")
    source.add_argument("--csv", type=Path, help="a curve CSV such as results/training_curve.csv")
    parser.add_argument("--out", type=Path, default=Path("outputs/morad/training_curve.png"))
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    rows = from_results(args.results) if args.results else from_csv(args.csv)
    if not rows:
        raise SystemExit("no evaluations found")

    csv_out = args.csv_out or (args.out.with_suffix(".csv") if args.results else None)
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", *COLUMNS])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_out}")

    steps = [r["step"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for column, label, marker in PLOTTED:
        values = [r[column] for r in rows]
        line, = ax.plot(steps, centred_mean(values, args.window), lw=2, label=label)
        ax.scatter(steps, values, s=12, marker=marker, color=line.get_color(), alpha=0.35, lw=0)
    ax.set_xlabel("Policy update")
    ax.set_ylabel("Held-out score (153 targets)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
