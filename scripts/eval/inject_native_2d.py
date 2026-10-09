"""Attach the native secondary structures to evaluation runs for the pairing metrics.

The pairing metrics (precision, recall, F1, MCC) compare the EternaFold
structure of each design with the base pairs of the native 3D structure. Those
reference pairs are distributed as test153_native_2d.json, keyed by target_id.
This script writes `evaluation_2d.json` next to each `evaluation.json`, with
`target_secondary_structure` filled in; the original file is left unchanged.

Usage
-----
    python scripts/eval/inject_native_2d.py \\
        --native-2d artifacts/pools/test153_native_2d.json \\
        --runs outputs/eval/test153/ride outputs/eval/test153/morad
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pick_dot_bracket(entry: dict) -> str | None:
    for key in ("db", "dot_bracket", "structure", "ss"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def inject(run_dir: Path, native: dict[str, dict]) -> tuple[int, int, int, int]:
    payload = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    targets = payload.get("targets") or []
    filled = length_mismatch = missing = 0
    for target in targets:
        entry = native.get(target.get("target_id"))
        dot_bracket = pick_dot_bracket(entry) if entry else None
        if not dot_bracket:
            missing += 1
            continue
        # a length mismatch would silently corrupt the pair comparison
        if len(dot_bracket) != len(target.get("native_sequence") or ""):
            length_mismatch += 1
            continue
        target["target_secondary_structure"] = dot_bracket
        target["target_secondary_source"] = "test153_native_2d.json"
        filled += 1
    (run_dir / "evaluation_2d.json").write_text(json.dumps(payload), encoding="utf-8")
    return filled, len(targets), length_mismatch, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--native-2d", default="artifacts/pools/test153_native_2d.json")
    parser.add_argument("--runs", nargs="+", required=True, help="evaluation output directories")
    args = parser.parse_args()

    native = json.loads(Path(args.native_2d).read_text(encoding="utf-8"))
    print(f"[native] {len(native)} reference secondary structures", flush=True)
    for run in args.runs:
        filled, total, mismatch, missing = inject(Path(run), native)
        print(
            f"  {run}: {filled}/{total} targets with native pairs "
            f"(length mismatch {mismatch}, no reference {missing})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
