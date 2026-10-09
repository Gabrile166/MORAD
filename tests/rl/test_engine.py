import copy
from dataclasses import dataclass

import torch

from src.rl.engine import RLEngine, RuntimeBudget
from src.rl.trainer import DiffusionNFTTrainer


@dataclass
class Target:
    target_id: str


@dataclass
class Scored:
    oracle_calls: int
    cache_summary: dict | None = None


class Rollout:
    def generate(self, target, condition, old_policy, state):
        return {"target_id": target.target_id}

    def state_dict(self):
        return {"temperature": 1.0}


class Rewarder:
    def score(self, rollout_batch, target):
        return Scored(oracle_calls=2, cache_summary={"fold_cache_misses": 2})


class Advantage:
    def compute(self, scored_batch):
        return object()

    def state_dict(self):
        return {"std": 1.0}


def loss(train_batch, current_policy, old_policy, reference_policy):
    value = current_policy(torch.ones(1, 1)).sum().pow(2)
    return {"loss_total": value}


def test_engine_runs_one_target_and_writes_checkpoints(tmp_path):
    current = torch.nn.Linear(1, 1)
    trainer = DiffusionNFTTrainer(
        current_policy=current,
        old_policy=copy.deepcopy(current),
        loss_fn=loss,
        config={"optim": {"lr": 0.01}},
        device="cpu",
    )
    engine = RLEngine(
        targets=[Target("t0")],
        rollout=Rollout(),
        rewarder=Rewarder(),
        advantage=Advantage(),
        trainer=trainer,
        budget=RuntimeBudget(max_outer_steps=1, max_oracle_calls=8, max_wall_hours=1, checkpoint_every_oracle_calls=2),
        checkpoint_dir=tmp_path,
        resolved_config={"ok": True},
        condition_factory=lambda target, **kwargs: object(),
    )
    result = engine.run()
    assert result["outer_step"] == 1
    assert result["oracle_calls"] == 2
    assert trainer.state.optimizer_step == 1
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "final.pt").exists()


class SkipAdvantage:
    def compute(self, scored_batch):
        return type("SkippedBatch", (), {"skip_update": True, "skip_reason": "metric_failed"})()

    def state_dict(self):
        return {}


def test_engine_skipped_result_includes_skip_reason_and_reward_summary():
    current = torch.nn.Linear(1, 1)
    trainer = DiffusionNFTTrainer(
        current_policy=current,
        old_policy=copy.deepcopy(current),
        loss_fn=loss,
        config={"optim": {"lr": 0.01}},
        device="cpu",
    )
    engine = RLEngine(
        targets=[Target("t0")],
        rollout=Rollout(),
        rewarder=Rewarder(),
        advantage=SkipAdvantage(),
        trainer=trainer,
        budget=RuntimeBudget(max_outer_steps=1),
        condition_factory=lambda target, **kwargs: object(),
    )

    result = engine.run_one_target(Target("t0"))

    assert result["status"] == "skipped"
    assert result["skip_reason"] == "metric_failed"
    assert result["reward"] == {"fold_cache_misses": 2}


def test_engine_eval_does_not_update_policy_optimizer_or_ema():
    current = torch.nn.Linear(1, 1)
    before = {name: value.detach().clone() for name, value in current.state_dict().items()}
    trainer = DiffusionNFTTrainer(
        current_policy=current,
        old_policy=copy.deepcopy(current),
        loss_fn=loss,
        config={"optim": {"lr": 0.01}},
        device="cpu",
    )
    engine = RLEngine(
        targets=[Target("t0")],
        rollout=Rollout(),
        rewarder=Rewarder(),
        advantage=Advantage(),
        trainer=trainer,
        budget=RuntimeBudget(max_outer_steps=1, max_oracle_calls=8, max_wall_hours=1),
        condition_factory=lambda target, **kwargs: object(),
    )

    result = engine.evaluate()

    assert result["mode"] == "eval"
    assert trainer.state.optimizer_step == 0
    assert trainer.state.ema_decay is None
    for name, value in current.state_dict().items():
        assert torch.equal(value, before[name])
