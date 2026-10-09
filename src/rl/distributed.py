"""Distributed (DDP) helpers for target-axis data-parallel RL training.

Launch model
------------
`torchrun --nproc_per_node=4 train.py ...` starts one process per GPU. Each
rank owns:

* one policy replica (its own CUDA device),
* one RhoFold+ oracle worker pinned to the same device,
* one target per step.

A step is therefore exactly one optimizer update over `world_size * group_size`
samples, which matches the conventional definition of a training step: one
batch, one parameter update.

Gradient synchronisation
------------------------
Every rank computes gradients for its own target's group and the gradients are
averaged across ranks before the optimizer step, so all replicas stay bitwise
identical. This is standard synchronous data parallelism -- the same objective
as accumulating those targets sequentially on one device, but every GPU does
real backward work instead of idling.

Because targets have different lengths, `DistributedDataParallel`'s automatic
bucketed all-reduce cannot be used directly (each rank traces a different graph
shape and DDP would complain about unused/mismatched parameters). We therefore
keep the module plain and all-reduce the gradient tensors explicitly, which is
mathematically the same operation DDP performs internally.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Iterable

import torch
import torch.distributed as dist


def is_torchrun() -> bool:
    """True when the process was launched by torchrun / torch.distributed."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def env_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def setup(timeout_minutes: int = 120) -> tuple[int, int, int]:
    """Initialise the process group. Returns ``(rank, local_rank, world_size)``.

    Safe to call when not under torchrun: it simply reports a single-process
    world so callers need no branching.
    """
    if not is_torchrun():
        return 0, 0, 1
    rank, local_rank, world = env_rank(), env_local_rank(), env_world_size()
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=_dt.timedelta(minutes=timeout_minutes),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def teardown() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if is_enabled() else 0


def world_size() -> int:
    return dist.get_world_size() if is_enabled() else 1


def is_main() -> bool:
    """Only rank 0 writes logs, wandb, and checkpoints."""
    return rank() == 0


def barrier() -> None:
    if is_enabled():
        dist.barrier()


def average_gradients(parameters: Iterable[torch.nn.Parameter], nonfinite_flag: bool = False) -> bool:
    """All-reduce gradients across ranks and divide by world size.

    Uses ONE fused all-reduce over a flat buffer instead of one call per
    parameter. That is what DistributedDataParallel does (bucketed), and it is
    what makes this correct under our usage: issuing hundreds of per-parameter
    collectives only works if every rank agrees, tensor for tensor, on which
    ones participate and in what order. `requires_grad` and `grad is None` are
    per-rank properties, and train_step / train_step_empty populate gradients
    differently, so filtering on them silently desynchronises the sequence and
    NCCL then busy-waits on mismatched operations forever.

    A single collective over a fixed-size buffer cannot desync: the shape and
    the call count are identical on every rank regardless of local state.
    """
    if not is_enabled():
        return bool(nonfinite_flag)
    # Materialise a stable, rank-independent list. No filtering on per-rank
    # state -- the parameter set comes from the module definition, so every
    # rank derives the same order and the same total size.
    params = [p for p in parameters]
    if not params:
        return bool(nonfinite_flag)
    ws = float(dist.get_world_size())
    grads = []
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        grads.append(param.grad)

    flat = torch._utils._flatten_dense_tensors(grads)
    # Carry the "did any rank see a bad loss" flag as one extra element of the
    # SAME buffer instead of a separate collective. Two separate all_reduce
    # calls of different sizes can be paired with each other by NCCL whenever
    # ranks drift by one operation, which hangs the job; a single call per step
    # makes that structurally impossible.
    payload = torch.cat([flat, torch.tensor([1.0 if nonfinite_flag else 0.0], device=flat.device, dtype=flat.dtype)])
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    flag_sum = float(payload[-1].item())
    flat = payload[:-1].div_(ws)
    for grad, reduced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        grad.copy_(reduced)
    return flag_sum > 0.0


def any_rank_true(flag: bool) -> bool:
    """Return True if `flag` is True on any rank -- a distributed logical OR.

    Used to keep control flow rank-symmetric: when only some ranks detect a bad
    batch, every rank still has to take the same branch, or the two groups issue
    different collectives and NCCL busy-waits on mismatched operations forever.
    Falls back to the local value when running without a process group.
    """
    if not is_enabled():
        return bool(flag)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item() > 0.0)


def broadcast_module(module: torch.nn.Module, src: int = 0) -> None:
    """Make every rank start from rank `src`'s exact weights."""
    if not is_enabled():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


def all_reduce_scalar(value: float, op: str = "mean") -> float:
    """Reduce a python float across ranks (for logging aggregate metrics)."""
    if not is_enabled():
        return float(value)
    device = torch.device("cuda", env_local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total = float(tensor.item())
    return total / float(dist.get_world_size()) if op == "mean" else total


def gather_object(obj: Any) -> list[Any]:
    """Gather arbitrary picklable objects from all ranks onto every rank."""
    if not is_enabled():
        return [obj]
    bucket: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(bucket, obj)
    return bucket


# ----------------------------------------------------------- target sub-groups
# When several ranks share one rollout target we need a collective spanning
# exactly those ranks (to merge that target's reward statistics) and no others.
_TARGET_GROUP = None
_TARGET_GROUP_RANKS: tuple[int, ...] = ()


def init_target_group(ranks_per_target: int) -> None:
    """Create the sub-group of ranks that share a rollout target.

    Must be called on *every* rank with the same argument: `new_group` is itself
    a collective, so all ranks have to walk the same creation sequence even for
    groups they do not join.
    """
    global _TARGET_GROUP, _TARGET_GROUP_RANKS
    if not is_enabled() or ranks_per_target <= 1:
        _TARGET_GROUP, _TARGET_GROUP_RANKS = None, ()
        return
    world = dist.get_world_size()
    if world % ranks_per_target != 0:
        raise ValueError(
            f"world_size {world} is not divisible by ranks_per_target {ranks_per_target}"
        )
    me = dist.get_rank()
    for start in range(0, world, ranks_per_target):
        members = tuple(range(start, start + ranks_per_target))
        group = dist.new_group(ranks=list(members))
        if me in members:
            _TARGET_GROUP, _TARGET_GROUP_RANKS = group, members


def target_group_ranks() -> tuple[int, ...]:
    return _TARGET_GROUP_RANKS


def target_id() -> int:
    """Index of the target this rank works on within the current batch."""
    if not is_enabled() or not _TARGET_GROUP_RANKS:
        return rank()
    return rank() // len(_TARGET_GROUP_RANKS)


def target_slot() -> int:
    """Position of this rank inside its target's rank set (0-based)."""
    if not is_enabled() or not _TARGET_GROUP_RANKS:
        return 0
    return rank() - _TARGET_GROUP_RANKS[0]


def gather_object_target_group(obj: Any) -> list[Any]:
    """Gather picklable objects from the ranks sharing this rank's target."""
    if _TARGET_GROUP is None:
        return [obj]
    bucket: list[Any] = [None] * len(_TARGET_GROUP_RANKS)
    dist.all_gather_object(bucket, obj, group=_TARGET_GROUP)
    return bucket


def merged_group_stats(local_rewards: list[float]) -> tuple[float, float, int]:
    """Merge reward statistics across the ranks sharing one target.

    Returns ``(mean, population_std, n)`` over the union of the participating
    ranks' finite rewards -- i.e. over the target's full rollout group, which is
    what GRPO requires. Falls back to local statistics when the target is not
    split across ranks.
    """
    import math

    if _TARGET_GROUP is None:
        finite = [float(r) for r in local_rewards if math.isfinite(float(r))]
    else:
        chunks = gather_object_target_group([float(r) for r in local_rewards])
        finite = [float(r) for chunk in chunks for r in chunk if math.isfinite(float(r))]
    n = len(finite)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(finite) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((r - mean) ** 2 for r in finite) / n
    return mean, math.sqrt(var), n
