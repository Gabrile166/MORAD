"""Shard a held-out validation across several GPUs and merge the summaries.

The evaluator folds one candidate at a time on a single device, so a full pass
over the 153 held-out targets takes tens of minutes. To keep a validation every
10 updates cheap relative to training, this splits the pool into per-GPU shards, runs one
evaluator process per shard concurrently, and pools the per-candidate metrics
back into one summary.

Sharding is round-robin over targets sorted by descending length: fold cost grows
with length, so dealing the longest targets out first keeps the shards balanced
rather than stranding one process on all the long chains.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def build_shards(
    pool_path: str | Path,
    shard_dir: str | Path,
    n_shards: int,
    *,
    prefix: str = "valshard",
) -> list[Path]:
    """Split a target pool into ``n_shards`` balanced pools, longest-first."""

    pool = torch.load(str(pool_path), map_location="cpu", weights_only=False)
    targets = list(pool["targets"])
    targets.sort(key=lambda t: -(t.get("metadata") or {}).get("length", 0))

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(n_shards):
        chunk = [t for position, t in enumerate(targets) if position % n_shards == index]
        shard = dict(pool)
        shard["targets"] = chunk
        for position, target in enumerate(chunk):
            metadata = target.get("metadata")
            if isinstance(metadata, dict):
                metadata = dict(metadata)
                metadata["dataset_index"] = position
                target = dict(target)
                target["metadata"] = metadata
                chunk[position] = target
        summary = shard.get("summary")
        if isinstance(summary, dict):
            shard["summary"] = {**summary, "total_targets": len(chunk)}
        path = shard_dir / f"{prefix}_{index}.pt"
        torch.save(shard, path)
        paths.append(path)
    return paths


def _pool_metric(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recombine per-shard metric blocks into one, weighting by candidate count.

    ``mean`` is re-derived from counts rather than averaged across shards, since
    the shards hold different numbers of candidates and averaging the averages
    would silently weight a 20-target shard the same as a 40-target one.
    """

    total_ok = sum(int(e.get("ok") or 0) for e in entries)
    total_count = sum(int(e.get("count") or 0) for e in entries)
    total_skipped = sum(int(e.get("skipped") or 0) for e in entries)
    total_error = sum(int(e.get("error") or 0) for e in entries)
    weighted = [
        (float(e["mean"]), int(e.get("ok") or 0))
        for e in entries
        if isinstance(e.get("mean"), (int, float)) and int(e.get("ok") or 0) > 0
    ]
    mean = sum(v * n for v, n in weighted) / sum(n for _, n in weighted) if weighted else None
    maxima = [float(e["max"]) for e in entries if isinstance(e.get("max"), (int, float))]
    minima = [float(e["min"]) for e in entries if isinstance(e.get("min"), (int, float))]
    return {
        "count": total_count,
        "ok": total_ok,
        "skipped": total_skipped,
        "error": total_error,
        "mean": mean,
        "max": max(maxima) if maxima else None,
        "min": min(minima) if minima else None,
    }


def merge_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge shard summaries, pooling every metric block they share."""

    merged: dict[str, Any] = {
        "target_count": sum(int(s.get("target_count") or 0) for s in summaries),
        "candidate_count": sum(int(s.get("candidate_count") or 0) for s in summaries),
        "shard_count": len(summaries),
    }
    for block in ("candidate_metrics", "target_metrics", "paper_single_sample_candidate_metrics"):
        names: set[str] = set()
        for summary in summaries:
            section = summary.get(block)
            if isinstance(section, Mapping):
                names |= set(section)
        if not names:
            continue
        pooled: dict[str, Any] = {}
        for name in sorted(names):
            entries = [
                summary[block][name]
                for summary in summaries
                if isinstance(summary.get(block), Mapping)
                and isinstance(summary[block].get(name), Mapping)
            ]
            if entries:
                pooled[name] = _pool_metric(entries)
        merged[block] = pooled
    return merged


_HEADLINE = (
    "sequence_recovery",
    "reward_c4p_gdt_ts",
    "reward_c4p_tm_score",
    "reward_c4p_rmsd",
    "rhofold_c1prime_tm_score",
    "reward_good",
    "secondary_structure_f1",
    "internal_diversity",
)


def _flatten_headline(merged: Mapping[str, Any]) -> dict[str, float]:
    """Lift the headline metric means to top-level scalars for wandb.

    ``paper_single_sample_candidate_metrics`` is the protocol's reported number
    (one candidate per target), so it wins where both blocks carry a metric.
    """

    flat: dict[str, float] = {}
    for block, prefix in (
        ("candidate_metrics", ""),
        ("target_metrics", ""),
        ("paper_single_sample_candidate_metrics", "paper_"),
    ):
        section = merged.get(block)
        if not isinstance(section, Mapping):
            continue
        for name in _HEADLINE:
            entry = section.get(name)
            if isinstance(entry, Mapping) and isinstance(entry.get("mean"), (int, float)):
                flat[f"{prefix}{name}"] = float(entry["mean"])
    return flat


class ShardedValidator:
    """Run a held-out evaluation across several GPUs, returning one summary.

    Mirrors ``PeriodicValidator``'s contract -- called with a step, returns a
    summary dict or ``None`` -- so the engine needs no special casing. Failures
    are surfaced as ``None`` and the engine treats validation as best-effort.
    """

    def __init__(
        self,
        *,
        repo_dir: str | os.PathLike[str],
        base_config: str,
        pool_path: str,
        devices: Sequence[str],
        output_root: str | os.PathLike[str],
        python_executable: str,
        rhofold_python: str,
        rhofold_checkpoint: str,
        usalign_binary: str,
        policy_provider: Any = None,
        n_samples: int = 8,
        timeout_sec: float = 5400.0,
        keep_runs: int = 3,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.base_config = base_config
        self.pool_path = pool_path
        self.devices = list(devices)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.python_executable = python_executable
        self.rhofold_python = rhofold_python
        self.rhofold_checkpoint = rhofold_checkpoint
        self.usalign_binary = usalign_binary
        self.policy_provider = policy_provider
        self.n_samples = int(n_samples)
        self.timeout_sec = float(timeout_sec)
        self.keep_runs = int(keep_runs)
        self.history: list[dict[str, Any]] = []
        # Async validation forks a fresh process per step, each with its own
        # validator, so self.history only holds that process's own entry.
        # This file is appended to by every worker and is the only
        # cross-process view of the run; retention must be based on it.
        self.results_path = self.output_root / "async_results.jsonl"
        self.shard_paths = build_shards(
            self.repo_dir / pool_path if not Path(pool_path).is_absolute() else pool_path,
            self.output_root / "shards",
            len(self.devices),
        )

    def _write_shard_config(self, shard_index: int, device: str, out_dir: Path) -> Path:
        import yaml

        with (self.repo_dir / self.base_config).open() as handle:
            config = yaml.safe_load(handle)
        shard_path = self.shard_paths[shard_index]
        config["ride"] = {
            **config.get("ride", {}),
            "pool_manifest": str(shard_path.relative_to(self.repo_dir))
            if shard_path.is_relative_to(self.repo_dir)
            else str(shard_path),
            "device": device,
        }
        config["sampling"] = {**config.get("sampling", {}), "n_samples": self.n_samples,
                              "target_limit": 100000}
        tertiary = dict(config.get("metrics", {}).get("tertiary", {}))
        tertiary.update(
            {
                "enabled": True,
                "device": device,
                "rhofold_python": self.rhofold_python,
                "checkpoint": self.rhofold_checkpoint,
                "usalign_binary": self.usalign_binary,
                "cache_dir": f"cache/val_shard{shard_index}",
            }
        )
        config["metrics"] = {**config.get("metrics", {}), "tertiary": tertiary}
        config["output"] = {"dir": str(out_dir)}
        path = out_dir / f"config_shard{shard_index}.yaml"
        with path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        return path

    def _snapshot_policy(self, step: int) -> str | None:
        if self.policy_provider is None:
            return None
        state = self.policy_provider()
        if state is None:
            return None
        path = self.output_root / f"policy_step{step:06d}.pt"
        torch.save({"model": state}, path)
        return str(path)

    # Metrics that decide which checkpoint is "best". RMSD is the only one where
    # lower is better, so it enters the score negated.
    _SCORE_KEYS: tuple[tuple[str, float], ...] = (
        ("paper_reward_c4p_gdt_ts", +1.0),
        ("paper_reward_c4p_tm_score", +1.0),
        ("paper_reward_c4p_rmsd", -1.0),
        ("paper_rhofold_c1prime_tm_score", +1.0),
    )

    def _load_history(self) -> list[dict[str, Any]]:
        """Every validation of this run, not just this process's own.

        Falls back to in-memory history when the shared file is absent so the
        synchronous path and unit tests keep working.
        """
        rows: list[dict[str, Any]] = []
        try:
            with self.results_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        # A worker killed mid-write leaves a torn line; one bad
                        # record must not blind us to the rest.
                        continue
                    if isinstance(entry, dict):
                        rows.append(entry)
        except OSError:
            pass
        by_step: dict[int, dict[str, Any]] = {}
        for entry in [*rows, *self.history]:
            step = entry.get("step")
            if isinstance(step, int):
                by_step[step] = entry
        return [by_step[s] for s in sorted(by_step)]

    def _composite_scores(self) -> dict[int, float]:
        """Rank the validated steps by a z-summed composite of four metrics.

        Standardising per metric first is what makes the sum meaningful: GDT and
        TM live in [0,1] while RMSD is in angstroms, so a raw sum would let RMSD
        dominate purely because of its scale. Steps missing any metric are left
        out rather than scored on partial evidence.
        """
        rows: list[tuple[int, list[float]]] = []
        for entry in self._load_history():
            step = entry.get("step")
            if not isinstance(step, int):
                continue
            vals = [entry.get(k) for k, _ in self._SCORE_KEYS]
            if all(isinstance(v, (int, float)) for v in vals):
                rows.append((step, [float(v) * sign
                                    for v, (_, sign) in zip(vals, self._SCORE_KEYS)]))
        if not rows:
            return {}
        n_metrics = len(self._SCORE_KEYS)
        means, stds = [], []
        for i in range(n_metrics):
            col = [r[1][i] for r in rows]
            mu = sum(col) / len(col)
            var = sum((c - mu) ** 2 for c in col) / len(col)
            means.append(mu)
            stds.append(var ** 0.5 or 1.0)
        return {
            step: sum((v - mu) / sd for v, mu, sd in zip(vals, means, stds))
            for step, vals in rows
        }

    def _steps_to_keep(self) -> set[int]:
        """Best + runner-up by composite score, plus the latest step.

        Purely time-ordered pruning threw away the best checkpoint of an entire
        run (the peak sits mid-training, not at the end), which left no way to
        evaluate the model at its best. Keeping the top two means a single noisy
        validation cannot cost us the good weights, and the latest is always kept
        so the final policy is available regardless of how it scored.
        """
        history = self._load_history()
        steps = [s for s in (e.get("step") for e in history) if isinstance(s, int)]
        if not steps:
            return set()
        # The three roles the user asked for -- best, runner-up, latest -- are
        # each unconditional. Seeding keep with the latest and then letting the
        # keep_runs budget stop the ranking loop meant that whenever the latest
        # step was not itself in the top two, one of best/runner-up got squeezed
        # out: that is how the runner-up snapshot (step 50, composite +2.59) was
        # deleted mid-run. The budget now only limits the discretionary top-up.
        scores = self._composite_scores()
        ranked = [s for s, _ in sorted(scores.items(), key=lambda kv: (-kv[1], -kv[0]))]
        keep = {max(steps)} | set(ranked[:2])
        # Discretionary top-up: fill any remaining budget with recent steps.
        for step in sorted(steps, reverse=True):
            if len(keep) >= max(len(keep), self.keep_runs):
                break
            keep.add(step)
        # Steps still being evaluated are absent from async_results.jsonl, so the
        # logic above cannot see them. Training outruns validation (a step takes
        # seconds, a full-pool validation ~410s), so several evaluations are
        # always in flight; without this their snapshots get deleted mid-run and
        # the subprocess dies on a missing file. These are exempt from the
        # keep_runs budget on purpose: the set is self-limiting because each
        # entry disappears as soon as its result lands.
        keep |= self._steps_in_flight()
        return keep

    def _steps_in_flight(self) -> set[int]:
        """Steps that have a dispatched shard directory but no result yet."""
        finished = {e.get("step") for e in self._load_history()}
        live: set[int] = set()
        for d in self.output_root.glob("step*"):
            if not d.is_dir():
                continue
            step = self._step_of(d)
            if step is not None and step not in finished:
                live.add(step)
        return live

    @staticmethod
    def _step_of(path: Path) -> int | None:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else None

    def _prune(self) -> None:
        """Bound disk use while never dropping the best-scoring checkpoint."""
        if self.keep_runs <= 0:
            return
        keep = self._steps_to_keep()
        if not keep:
            return
        for stale in sorted(p for p in self.output_root.glob("step*") if p.is_dir()):
            step = self._step_of(stale)
            if step is None or step in keep:
                continue
            for child in sorted(stale.rglob("*"), reverse=True):
                try:
                    child.unlink() if child.is_file() else child.rmdir()
                except OSError:
                    pass
            try:
                stale.rmdir()
            except OSError:
                pass
        for stale_ckpt in sorted(self.output_root.glob("policy_step*.pt")):
            step = self._step_of(stale_ckpt)
            if step is None or step in keep:
                continue
            try:
                stale_ckpt.unlink()
            except OSError:
                pass
        try:
            scores = self._composite_scores()
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], -kv[0]))
            (self.output_root / "kept_checkpoints.json").write_text(json.dumps({
                "kept_steps": sorted(keep),
                "score_keys": [k for k, _ in self._SCORE_KEYS],
                "ranking": [{"step": s, "composite_z": round(z, 6)} for s, z in ranked],
            }, indent=1))
        except OSError:
            pass

    def run_snapshot(self, *, snapshot: Path, step: int, tag: str = "periodic") -> dict[str, Any] | None:
        """Score a policy snapshot that was already written to disk.

        Used by the detached validation worker: the training process only dumps
        the state_dict, and scoring happens here, out of the training loop, so no
        rank is ever blocked long enough to break the NCCL group.

        Reuses __call__ by pointing policy_provider at the saved file instead of
        a live module, so both paths share one code path.
        """
        previous = self.policy_provider
        saved = str(snapshot)

        def _from_disk():
            return None  # suppress re-snapshotting; we set RIDE_CHECKPOINT below

        self.policy_provider = _from_disk
        prior_env = os.environ.get("RIDE_CHECKPOINT")
        os.environ["RIDE_CHECKPOINT"] = saved
        try:
            return self(step=step, tag=tag)
        finally:
            self.policy_provider = previous
            if prior_env is None:
                os.environ.pop("RIDE_CHECKPOINT", None)
            else:
                os.environ["RIDE_CHECKPOINT"] = prior_env

    def __call__(self, step: int, tag: str = "periodic", engine: Any = None,
                 **_: Any) -> dict[str, Any] | None:
        """Run one validation. `tag`/`engine` come from the engine's call site."""
        out_dir = self.output_root / f"step{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        checkpoint = self._snapshot_policy(step)
        if checkpoint:
            env["RIDE_CHECKPOINT"] = checkpoint
        # else: keep whatever RIDE_CHECKPOINT the caller already put in the
        # environment (run_snapshot does exactly that for an on-disk snapshot).
        # torchrun exports these; a child that inherits them tries to join the
        # training process group and hangs before it ever folds anything.
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "TORCHELASTIC_RUN_ID"):
            env.pop(key, None)

        started = time.time()
        procs = []
        for index, device in enumerate(self.devices):
            shard_out = out_dir / f"shard{index}"
            shard_out.mkdir(parents=True, exist_ok=True)
            config_path = self._write_shard_config(index, device, shard_out)
            procs.append(
                (
                    index,
                    shard_out,
                    subprocess.Popen(
                        [
                            self.python_executable,
                            "evaluate.py",
                            "--config",
                            str(config_path),
                            "--output-dir",
                            str(shard_out),
                        ],
                        cwd=str(self.repo_dir),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    ),
                )
            )

        summaries: list[dict[str, Any]] = []
        failures: list[str] = []
        for index, shard_out, proc in procs:
            remaining = max(30.0, self.timeout_sec - (time.time() - started))
            try:
                stdout, stderr = proc.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                failures.append(f"shard{index}: timeout")
                continue
            (shard_out / "stdout.log").write_text(stdout or "", encoding="utf-8")
            (shard_out / "stderr.log").write_text(stderr or "", encoding="utf-8")
            summary_path = shard_out / "summary.json"
            if proc.returncode != 0 or not summary_path.exists():
                failures.append(f"shard{index}: rc={proc.returncode}")
                continue
            summaries.append(json.loads(summary_path.read_text()))

        if not summaries:
            (out_dir / "validation_error.txt").write_text("\n".join(failures) or "no summaries\n")
            return None

        merged = merge_summaries(summaries)
        merged["elapsed_sec"] = time.time() - started
        merged["failed_shards"] = failures
        # The engine forwards only top-level scalars to wandb, so lift the
        # headline metrics out of the nested blocks where they'd be invisible.
        merged.update(_flatten_headline(merged))
        (out_dir / "summary.json").write_text(json.dumps(merged, indent=1, sort_keys=True))
        self.history.append({"step": step, **merged})
        self._prune()
        return merged