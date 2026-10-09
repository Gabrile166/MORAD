"""Fold every designed candidate with RhoFold+ for the lDDT / pLDDT / clash panel.

evaluate.py folds candidate 0 of each target. This script folds the remaining
candidates of one or more evaluation runs with one persistent RhoFold+ worker
per GPU. Identical sequences are folded once, and results are cached on disk
(keyed by sequence hash) so reruns and additional runs are cheap.

Usage
-----
    python scripts/eval/fold_all_candidates.py \\
        --runs ride=outputs/eval/test153/ride,morad=outputs/eval/test153/morad \\
        --cache outputs/eval/test153/fold_cache.json \\
        --scratch outputs/eval/test153/fold_scratch --gpus 0,1,2,3

RHOFOLD_PYTHON and RHOFOLD_CHECKPOINT are read from the environment
(see scripts/env.example.sh).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def parse_runs(spec: str) -> list[tuple[str, Path]]:
    """`name=dir,name=dir` -> [(name, dir)]; each dir holds an evaluation.json."""
    runs = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, path = chunk.partition("=")
        runs.append((name.strip(), Path(path.strip())))
    return runs


class FoldWorker:
    """One persistent RhoFold+ process pinned to one GPU."""

    def __init__(self, gpu: int, python: str, checkpoint: str, repo: Path, scratch: Path):
        self.gpu = gpu
        self.scratch = scratch / f"gpu{gpu}"
        self.scratch.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        self.process = subprocess.Popen(
            [
                python,
                "scripts/rhofold_oracle_worker.py",
                "--ckpt",
                checkpoint,
                "--device",
                "cuda",
            ],
            cwd=str(repo),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            bufsize=1,
        )

    def fold(self, request_id: str, sequence: str) -> dict[str, Any]:
        payload = {
            "request_id": request_id,
            "sequence": sequence,
            "output_dir": str(self.scratch / request_id),
        }
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            return {"error_message": "worker died"}
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            return {"error_message": f"bad json: {exc}"}

    def close(self) -> None:
        try:
            self.process.stdin.close()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


def load_jobs(runs: list[tuple[str, Path]], cache: dict) -> list[dict]:
    """Every unique candidate sequence that still needs folding."""
    jobs: dict[str, dict] = {}
    for _, run_dir in runs:
        payload = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        for entry in payload.get("candidates", []):
            sequence = entry.get("sequence")
            if not sequence:
                continue
            key = sequence_key(sequence)
            if key in cache:
                continue
            jobs.setdefault(key, {"key": key, "sequence": sequence, "length": len(sequence)})
    # shortest first: fills the pipeline fast and surfaces failures early
    return sorted(jobs.values(), key=lambda j: j["length"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", required=True, help="comma-separated name=evaluation_dir entries")
    parser.add_argument("--cache", required=True, help="JSON cache of folded sequences")
    parser.add_argument("--scratch", required=True, help="directory for predicted structures")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    cache_path = Path(args.cache)
    scratch = Path(args.scratch).resolve()
    python = os.environ.get("RHOFOLD_PYTHON", "python")
    checkpoint = os.environ.get(
        "RHOFOLD_CHECKPOINT",
        str(repo.parent / "third_party/rhofold_protocol/checkpoints/rhofold_pretrained_params.pt"),
    )

    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[cache] {len(cache)} sequences already folded", flush=True)

    runs = parse_runs(args.runs)
    jobs = load_jobs(runs, cache)
    jobs = [j for j in jobs if j["length"] <= args.max_length]
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print("[jobs] nothing to fold", flush=True)
        return 0
    print(
        f"[jobs] {len(jobs)} sequences to fold (lengths {jobs[0]['length']}..{jobs[-1]['length']})",
        flush=True,
    )

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    print(f"[workers] starting {len(gpus)} RhoFold+ workers on GPUs {gpus}", flush=True)
    workers = [FoldWorker(g, python, checkpoint, repo, scratch) for g in gpus]

    work_queue: queue.Queue = queue.Queue()
    for job in jobs:
        work_queue.put(job)

    lock = threading.Lock()
    done = {"count": 0, "failed": 0}
    started = time.time()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def run(worker: FoldWorker) -> None:
        while True:
            try:
                job = work_queue.get_nowait()
            except queue.Empty:
                return
            response = worker.fold(job["key"], job["sequence"])
            error = response.get("error_message")
            record = {
                "length": job["length"],
                "plddt": response.get("plddt"),
                "structure_path": response.get("predicted_structure_path"),
                "structure_hash": response.get("predicted_structure_hash"),
                "error": error,
            }
            with lock:
                cache[job["key"]] = record
                done["count"] += 1
                if error:
                    done["failed"] += 1
                if done["count"] % 25 == 0:
                    elapsed = time.time() - started
                    rate = done["count"] / elapsed
                    remaining = (len(jobs) - done["count"]) / max(rate, 1e-9)
                    print(
                        f"  {done['count']}/{len(jobs)}  failed={done['failed']}  "
                        f"{rate:.2f} seq/s  eta {remaining / 60:.1f} min",
                        flush=True,
                    )
                    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    threads = [threading.Thread(target=run, args=(w,), daemon=True) for w in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for worker in workers:
        worker.close()

    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    elapsed = time.time() - started
    print(
        f"[done] folded {done['count']} sequences in {elapsed / 60:.1f} min "
        f"({done['failed']} failed); cache at {cache_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
