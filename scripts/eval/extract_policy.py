#!/usr/bin/env python
"""Extract the policy weights from a full training checkpoint.

The result is saved as {"model": state_dict}, the format written for the
validation snapshots and read by evaluate.py through RIDE_CHECKPOINT.

Usage: python scripts/eval/extract_policy.py <checkpoint.pt> <out.pt> [--key current_policy]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

# training checkpoints pickle types from src.*, so the repository must be importable
_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("dst", type=pathlib.Path)
    ap.add_argument("--key", default="current_policy", help="state-dict key to extract")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        sys.exit(f"[!] {args.src} is not a checkpoint dictionary")

    sd = ck.get(args.key)
    if not isinstance(sd, dict) or not sd:
        sys.exit(f"[!] key {args.key!r} is missing or empty; available keys: {list(ck.keys())[:12]}")

    # keep tensors only and drop a possible DDP "module." prefix
    clean = {}
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        clean[k[len("module."):] if k.startswith("module.") else k] = v

    if not clean:
        sys.exit(f"[!] key {args.key!r} holds no tensors")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": clean}, args.dst)

    cursor = ck.get("cursor", {})
    print(f"  source: {args.src}")
    print(f"  key: {args.key}  tensors: {len(clean)}")
    if isinstance(cursor, dict) and cursor:
        print(f"  training cursor: {dict(list(cursor.items())[:5])}")
    print(f"  wrote: {args.dst}")


if __name__ == "__main__":
    main()
