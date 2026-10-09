"""Multi-GPU RhoFold+ oracle pool.

Profiling of the v1 run showed where an outer step actually goes:

    condition            0.01s   ( 0.1%)
    rollout_and_score   22.07s   (94.5%)   <-- 8 sequential RhoFold folds
    advantage            0.00s   ( 0.0%)
    train_step           1.27s   ( 5.5%)

`score_rollout_batch` folds the `group_size` candidates one at a time through a
single worker subprocess pinned to one GPU. With 5 usable GPUs the fold phase is
embarrassingly parallel: each candidate is an independent forward pass.

This pool keeps N persistent workers, each pinned to its own GPU, and exposes the
same `fold()` signature as `RhoFoldJsonlClient` (so it is a drop-in replacement),
plus `fold_many()` which dispatches a whole group concurrently.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence


class RhoFoldPool:
    """Round-robin pool of single-GPU RhoFold+ worker clients.

    Each underlying client owns one subprocess pinned to one GPU via
    CUDA_VISIBLE_DEVICES, so folds on different GPUs proceed in true parallel
    (the GIL is released while blocking on subprocess I/O).
    """

    def __init__(
        self,
        command: Sequence[str],
        gpu_uuids: Sequence[str],
        *,
        timeout_s: float = 300.0,
        output_root: str | Path | None = None,
        device_arg: str = "cuda:0",
    ) -> None:
        from src.rl.reward import RhoFoldJsonlClient

        if not gpu_uuids:
            raise ValueError("RhoFoldPool requires at least one GPU uuid")
        self.gpu_uuids = list(gpu_uuids)
        self.output_root = Path(output_root) if output_root is not None else None
        self._lock = threading.Lock()
        self._cursor = 0
        self.clients: list[RhoFoldJsonlClient] = []

        base_env = dict(os.environ)
        for uuid in self.gpu_uuids:
            # Each worker sees exactly one GPU, which it addresses as cuda:0.
            # Pinning by UUID avoids index/CUDA-ordinal mismatches on hosts
            # where some physical GPUs are unavailable.
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            os.environ["RHOFOLD_DEVICE"] = device_arg
            try:
                client = RhoFoldJsonlClient(
                    command=list(command),
                    timeout_s=timeout_s,
                    output_root=self.output_root,
                )
            finally:
                os.environ.clear()
                os.environ.update(base_env)
            self.clients.append(client)

        self._executor = ThreadPoolExecutor(
            max_workers=len(self.clients), thread_name_prefix="rhofold-pool"
        )

    # --------------------------------------------------------------- interface
    @property
    def size(self) -> int:
        return len(self.clients)

    def _next_client(self):
        with self._lock:
            client = self.clients[self._cursor % len(self.clients)]
            self._cursor += 1
        return client

    def fold(self, sequence: str, output_dir: str | Path, target_id: str | None = None):
        """Single fold, kept for drop-in compatibility with RhoFoldJsonlClient."""
        return self._next_client().fold(sequence, output_dir, target_id=target_id)

    def fold_many(
        self,
        sequences: Sequence[str],
        output_dirs: Sequence[str | Path],
        target_id: str | None = None,
    ) -> list:
        """Fold a whole group concurrently, one sequence per worker/GPU.

        Results are returned in the same order as `sequences`. A worker failure
        is surfaced as its FoldResult rather than raising, so one bad candidate
        cannot abort the step.
        """
        if len(sequences) != len(output_dirs):
            raise ValueError("sequences and output_dirs must have equal length")
        if not sequences:
            return []

        assignments = [
            (self.clients[i % len(self.clients)], seq, out)
            for i, (seq, out) in enumerate(zip(sequences, output_dirs))
        ]

        def _run(job):
            client, seq, out = job
            return client.fold(seq, out, target_id=target_id)

        return list(self._executor.map(_run, assignments))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        for client in self.clients:
            try:
                client.close()
            except Exception:
                pass
