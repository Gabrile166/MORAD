"""MORAD training entry point: online RL post-training of a pretrained RIDE policy."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.rl import distributed as dist_utils
from src.rl.checkpoint import load_checkpoint, write_manifest
from src.rl.config import RLConfig, load_config
from src.rl.engine import RLEngine, RuntimeBudget
from src.rl.observability import JsonlLogger, build_run_manifest
from src.rl.runtime import build_noise_scheduler, load_ride_model
from src.rl.trainer import DiffusionNFTTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _du_rank0() -> bool:
    """True only on rank 0 (or when not running under torchrun)."""
    return dist_utils.is_main()


class _DevTinyPolicy(torch.nn.Module):
    out_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def encode_condition(self, condition: Any) -> torch.Tensor:
        seq = getattr(condition, "seq", torch.arange(1))
        return seq.float().sum().reshape(())

    def predict_noise_from_encoding(self, encoding: Any, z_t: torch.Tensor, noise_level: torch.Tensor, time: torch.Tensor | None = None) -> torch.Tensor:
        return z_t * torch.tanh(self.weight) + noise_level.reshape(-1, 1, 1) * 0.0


def build_real_engine(config: RLConfig, output_dir: Path, mode: str) -> RLEngine:
    """Wire production collaborators when all RL modules are present.

    Under torchrun each rank builds its own engine on its own GPU. The oracle
    worker is pinned to that GPU so 8 ranks do not all contend for cuda:0. Only
    rank 0 writes the manifest and initialises wandb; the other ranks get a null
    logger, which the engine interprets as "log nothing".
    """
    from src.rl.advantage import AdvantageComputer
    from src.rl.reward import RewardScorer
    from src.rl.rollout import AdaptiveX0RenoiseRollout
    from src.rl.targets import load_target_pool

    cfg = config.to_dict()
    local_rank = dist_utils.env_local_rank() if dist_utils.is_torchrun() else 0
    if dist_utils.is_torchrun() and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    current_policy = _DevTinyPolicy().to(device) if config.runtime.dev_tiny_policy else load_ride_model(config, device)
    old_policy = copy.deepcopy(current_policy)
    reference_policy = copy.deepcopy(current_policy) if config.trainer.loss.reference_weight else None
    trainer = DiffusionNFTTrainer(
        current_policy=current_policy,
        old_policy=old_policy,
        reference_policy=reference_policy,
        config=cfg["trainer"],
        device=device,
    )

    # Pin this rank's oracle worker to its own GPU.
    reward_cfg = dict(cfg.get("reward", {}))
    oracle_cfg = dict(reward_cfg.get("oracle", {}))
    oracle_cfg["device"] = str(device)
    oracle_cfg.setdefault("python", os.environ.get("RHOFOLD_PYTHON", "python"))
    reward_cfg["oracle"] = oracle_cfg
    rewarder = RewardScorer(reward_cfg)

    from src.rl.observability import maybe_init_wandb

    is_main = _du_rank0()
    logger = None
    manifest = {}
    if is_main:
        logger = JsonlLogger(output_dir / "events.jsonl", wandb_run=maybe_init_wandb(cfg.get("logging", {}), cfg))
        manifest = build_run_manifest(Path(__file__).resolve().parent, cfg, extra={"oracle": rewarder.manifest_state(), "mode": mode})
        write_manifest(output_dir / "run_manifest.json", manifest)

    targets = load_target_pool(config.data.pool_manifest)
    # Random order instead of the curriculum's easy-to-hard sequence: with a
    # batch of targets a sorted order makes each batch homogeneous in length, so
    # the gradient tracks the curriculum stage rather than the task.
    shuffle = bool(cfg.get("runtime", {}).get("shuffle_targets", False))
    shuffle_seed = int(cfg.get("runtime", {}).get("shuffle_seed", 0))
    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(targets)
        if is_main:
            print(f"[train] shuffled {len(targets)} targets (seed={shuffle_seed})", flush=True)

    # The engine walks the target list once (`total_steps = len(targets) //
    # batch_targets`), so a second epoch means appending a second pass rather
    # than raising max_outer_steps -- which would just be clamped. Each extra
    # pass is reshuffled so the batches differ between epochs.
    epochs = max(1, int(cfg.get("runtime", {}).get("epochs", 1)))
    if epochs > 1:
        one_epoch = list(targets)
        for epoch in range(1, epochs):
            extra = list(one_epoch)
            if shuffle:
                random.Random(shuffle_seed + epoch).shuffle(extra)
            targets.extend(extra)
        if is_main:
            print(
                f"[train] {epochs} epochs -> {len(targets)} target visits "
                f"({len(one_epoch)} unique)",
                flush=True,
            )

    # Held-out validation. The engine already knows how to call this every
    # `test_freq` updates and once before training, and it barriers afterwards so
    # the non-main ranks stay in step; it just needs to be handed the callable.
    val_fn = None
    val_cfg = dict(cfg.get("validation", {}) or {})
    if is_main and bool(val_cfg.get("enabled", False)):
        from src.rl.sharded_validation import ShardedValidator

        # Fold cost is the bottleneck (measured 14s/target), so the validation
        # fans out over whichever GPUs are free rather than folding serially on
        # one. Training holds ranks 0..world-1; anything above that is idle while
        # rank 0 blocks here, so those are what the shards run on.
        devices = [d.strip() for d in str(val_cfg.get("devices", "")).split(",") if d.strip()]
        if not devices:
            devices = [str(device)]
        validator = ShardedValidator(
            repo_dir=Path(__file__).resolve().parent,
            base_config=str(val_cfg.get("config", "")),
            pool_path=str(val_cfg.get("pool_manifest", "")),
            devices=devices,
            output_root=output_dir / "validation",
            python_executable=sys.executable,
            rhofold_python=os.environ.get("RHOFOLD_PYTHON", "python"),
            rhofold_checkpoint=os.environ.get(
                "RHOFOLD_CHECKPOINT", "../third_party/rhofold_protocol/checkpoints/rhofold_pretrained_params.pt"
            ),
            usalign_binary=str(val_cfg.get("usalign_binary", "../third_party/USalign/USalign")),
            policy_provider=lambda: {
                k: v.detach().cpu().clone() for k, v in trainer.current_policy.state_dict().items()
            },
            n_samples=int(val_cfg.get("n_samples", 8)),
            timeout_sec=float(val_cfg.get("timeout_sec", 5400.0)),
            keep_runs=int(val_cfg.get("keep_runs", 3)),
        )
        val_fn = validator
        print(
            f"[validation] enabled: pool={val_cfg.get('pool_manifest')} "
            f"devices={devices} test_freq={val_cfg.get('test_freq')} "
            f"before_train={val_cfg.get('before_train')} "
            f"shards={[p.name for p in validator.shard_paths]}",
            flush=True,
        )

    return RLEngine(
        targets=targets,
        rollout=AdaptiveX0RenoiseRollout(build_noise_scheduler(config), cfg["rollout"]),
        rewarder=rewarder,
        advantage=AdvantageComputer(cfg["advantage"]),
        trainer=trainer,
        old_policy=old_policy,
        budget=RuntimeBudget(**cfg["runtime"]["budget"]),
        logger=logger,
        checkpoint_dir=output_dir / "checkpoints",
        resolved_config=cfg,
        manifest=manifest,
        condition_noise_scale=config.data.condition_noise_scale,
        device=str(device),
        val_fn=val_fn,
        test_freq=int(val_cfg.get("test_freq", 0)),
        val_before_train=bool(val_cfg.get("before_train", False)),
    )


@dataclass
class _FakeTarget:
    target_id: str


@dataclass
class _FakeScored:
    oracle_calls: int = 1


class _FakeRollout:
    def generate(self, target: _FakeTarget, condition: Any, old_policy: Any, state: Any) -> dict[str, Any]:
        return {"target_id": target.target_id, "samples": [0, 1]}

    def state_dict(self) -> dict[str, Any]:
        return {"temperature": 1.0}


class _FakeRewarder:
    def score(self, rollout_batch: Mapping[str, Any], target: _FakeTarget) -> _FakeScored:
        return _FakeScored(oracle_calls=len(rollout_batch["samples"]))


class _FakeAdvantage:
    def compute(self, scored_batch: _FakeScored) -> object:
        return object()

    def state_dict(self) -> dict[str, Any]:
        return {"reward_scale": 1.0}


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)

    def forward(self) -> torch.Tensor:
        return self.layer(torch.eye(4)).mean()


def _smoke_loss(train_batch: Any, current_policy: torch.nn.Module, old_policy: torch.nn.Module, reference_policy: torch.nn.Module | None):
    loss = current_policy().pow(2)
    return {"loss_total": loss, "loss_policy": loss.detach() + 0.0}


def build_smoke_engine(config: Mapping[str, Any], output_dir: Path) -> RLEngine:
    current = _TinyPolicy()
    old = copy.deepcopy(current)
    reference = copy.deepcopy(current)
    trainer = DiffusionNFTTrainer(
        current_policy=current,
        old_policy=old,
        reference_policy=reference,
        loss_fn=_smoke_loss,
        config=config.get("trainer", {}),
        device="cpu",
    )
    logger = JsonlLogger(output_dir / "events.jsonl")
    manifest = build_run_manifest(Path(__file__).resolve().parent, config, command=sys.argv, extra={"mode": "smoke"})
    write_manifest(output_dir / "run_manifest.json", manifest)
    return RLEngine(
        targets=[_FakeTarget("smoke_target")],
        rollout=_FakeRollout(),
        rewarder=_FakeRewarder(),
        advantage=_FakeAdvantage(),
        trainer=trainer,
        budget=RuntimeBudget(max_outer_steps=1, max_oracle_calls=8, max_wall_hours=0.1, checkpoint_every_oracle_calls=1),
        logger=logger,
        checkpoint_dir=output_dir / "checkpoints",
        resolved_config=config,
        manifest=manifest,
        condition_factory=lambda target, **kwargs: object(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/morad.yaml")
    parser.add_argument("--mode", choices=["train", "resume", "eval", "smoke"], default="smoke")
    parser.add_argument("--output-dir", default="outputs/morad")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # The out-of-process validation worker must read THIS run's config, so that
    # its fold workers land on this run's validation.devices.
    os.environ["RIDE_RL_CONFIG"] = str(args.config)

    # Initialise the process group before touching any GPU or building any
    # module that might need collective communication.
    if dist_utils.is_torchrun():
        dist_utils.setup()

    config = load_config(args.config)
    if args.seed is not None:
        config = RLConfig(**{**config.__dict__, "runtime": type(config.runtime)(**{**config.runtime.__dict__, "seed": args.seed})}).validate()
    if args.mode != "smoke":
        config = RLConfig(**{**config.__dict__, "runtime": type(config.runtime)(**{**config.runtime.__dict__, "mode": "train" if args.mode == "resume" else args.mode})}).validate()
    cfg_dict = config.to_dict()
    set_seed(int(config.runtime.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = build_smoke_engine(cfg_dict, output_dir) if args.mode == "smoke" else build_real_engine(config, output_dir, args.mode)
    if args.mode == "resume":
        if not args.resume:
            raise ValueError("--resume is required when --mode resume")
        payload = load_checkpoint(
            args.resume,
            current_policy=engine.trainer.current_policy,
            old_policy=engine.trainer.old_policy,
            reference_policy=engine.trainer.reference_policy,
            optimizer=engine.trainer.optimizer,
            amp_scaler=getattr(engine.trainer, "scaler", None),
            expected_config=cfg_dict,
            map_location=getattr(engine.trainer, "device", "cpu"),
        )
        engine.load_state_dict({**payload.get("extra_state", {}), "reward_scale_state": payload.get("reward_scale_state", {}), "temperature_state": payload.get("temperature_state", {})})
    result = engine.evaluate() if args.mode == "eval" else engine.run()
    if args.mode == "eval":
        (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))

    if dist_utils.is_enabled():
        dist_utils.teardown()


if __name__ == "__main__":
    main()
