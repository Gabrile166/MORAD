#!/usr/bin/env python
"""Score one policy snapshot off the training critical path.

Launched detached by RLEngine._enqueue_async_validation. Runs the same sharded
validation as before, but in its own process, so the training ranks are never
blocked and the NCCL group is never left waiting on a rank busy folding RNA.

Writes <out>/step<NNNNNN>/summary.json plus a line in <out>/async_results.jsonl,
which the reporting step reads to build the validation curve.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--tag", default="periodic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    # Never inherit the trainer's rendezvous, or this process joins its group.
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
              "GROUP_RANK", "ROLE_RANK", "LOCAL_WORLD_SIZE",
              "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID"):
        os.environ.pop(k, None)

    import yaml
    from src.rl.sharded_validation import ShardedValidator

    with (REPO / args.config).open() as fh:
        cfg = yaml.safe_load(fh)
    val = cfg.get("validation", {})
    devices = [d.strip() for d in str(val.get("devices", "")).split(",") if d.strip()]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    validator = ShardedValidator(
        repo_dir=REPO,
        base_config=str(val.get("config", "")),
        pool_path=str(val.get("pool_manifest", "")),
        devices=devices,
        output_root=out,
        python_executable=sys.executable,
        rhofold_python=os.environ.get("RHOFOLD_PYTHON", "python"),
        rhofold_checkpoint=os.environ.get(
            "RHOFOLD_CHECKPOINT", "../third_party/rhofold_protocol/checkpoints/rhofold_pretrained_params.pt"
        ),
        usalign_binary=str(val.get("usalign_binary", "../third_party/USalign/USalign")),
        policy_provider=None,          # snapshot already on disk
        n_samples=int(val.get("n_samples", 8)),
        timeout_sec=float(val.get("timeout_sec", 5400.0)),
        keep_runs=int(val.get("keep_runs", 3)),
    )
    summary = validator.run_snapshot(snapshot=Path(args.snapshot), step=args.step, tag=args.tag)
    elapsed = time.perf_counter() - started

    rec = {"step": args.step, "tag": args.tag, "elapsed_sec": elapsed,
           "finished_at": time.time(), **(summary or {})}
    with (out / "async_results.jsonl").open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print(f"[async-eval] step={args.step} finished in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())