#!/usr/bin/env python
"""Run the standalone evaluation protocol on a RIDE/MORAD policy or on precomputed baseline FASTAs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.config import load_evaluation_config
from src.evaluation.runner import RiboDiffusionEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval/test153.yaml")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config = load_evaluation_config(args.config)
    output_dir = args.output_dir or config.get("output", {}).get("dir", "outputs/eval/test153")
    evaluator = RiboDiffusionEvaluator(config, repo_root=repo_root, output_dir=repo_root / output_dir)
    result = evaluator.run()
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
