import copy
from dataclasses import dataclass

import torch

from src.rl.trainer import DiffusionNFTTrainer


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1, bias=False)

    def forward(self):
        return self.layer(torch.ones(1, 2)).sum()


def finite_loss(train_batch, current_policy, old_policy, reference_policy):
    loss = current_policy().pow(2)
    return {"loss_total": loss, "loss_policy": loss}


def nan_loss(train_batch, current_policy, old_policy, reference_policy):
    return {"loss_total": current_policy() * torch.tensor(float("nan"))}


def test_trainer_eval_modes_grad_and_ema_update():
    current = TinyPolicy()
    old = copy.deepcopy(current)
    trainer = DiffusionNFTTrainer(
        current_policy=current,
        old_policy=old,
        reference_policy=copy.deepcopy(current),
        loss_fn=finite_loss,
        config={"optim": {"lr": 0.1, "grad_clip_norm": 10.0}, "ema": {"decay_max": 0.99, "update_interval": 1}},
        device="cpu",
    )
    before_old = next(trainer.old_policy.parameters()).detach().clone()
    result = trainer.train_step(object())
    assert result["status"] == "ok"
    assert trainer.current_policy.training is False
    assert trainer.old_policy.training is False
    assert next(trainer.current_policy.parameters()).requires_grad
    assert not next(trainer.old_policy.parameters()).requires_grad
    assert trainer.state.optimizer_step == 1
    assert trainer.state.ema_decay == min(2 / 11, 0.99)
    assert not torch.equal(before_old, next(trainer.old_policy.parameters()).detach())


def test_trainer_skips_nonfinite_loss_without_step():
    current = TinyPolicy()
    trainer = DiffusionNFTTrainer(
        current_policy=current,
        old_policy=copy.deepcopy(current),
        loss_fn=nan_loss,
        config={"optim": {"lr": 0.1}},
        device="cpu",
    )
    result = trainer.train_step(object())
    assert result["status"] == "skipped_nonfinite_loss"
    assert trainer.state.optimizer_step == 0
    assert trainer.state.skipped_steps == 1


@dataclass(frozen=True)
class Batch:
    x0_onehot: torch.Tensor
    target: torch.Tensor

    def select(self, mask):
        return Batch(self.x0_onehot[mask], self.target[mask])


class BatchPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.layer(x).squeeze(-1)


def batch_loss(train_batch, current_policy, old_policy, reference_policy):
    pred = current_policy(train_batch.x0_onehot)
    loss = (pred - train_batch.target).square().mean()
    return {"loss_total": loss, "loss_policy": loss}


def test_microbatching_matches_full_batch_update():
    torch.manual_seed(0)
    batch = Batch(torch.randn(4, 2), torch.randn(4))
    full_policy = BatchPolicy()
    micro_policy = copy.deepcopy(full_policy)
    full = DiffusionNFTTrainer(
        current_policy=full_policy,
        old_policy=copy.deepcopy(full_policy),
        loss_fn=batch_loss,
        config={"optim": {"lr": 0.05, "grad_clip_norm": 100.0, "microbatch_size": 99}},
        device="cpu",
    )
    micro = DiffusionNFTTrainer(
        current_policy=micro_policy,
        old_policy=copy.deepcopy(micro_policy),
        loss_fn=batch_loss,
        config={"optim": {"lr": 0.05, "grad_clip_norm": 100.0, "microbatch_size": 1}},
        device="cpu",
    )

    full.train_step(batch)
    micro.train_step(batch)

    assert full.state.optimizer_step == micro.state.optimizer_step == 1
    assert torch.allclose(full.current_policy.layer.weight, micro.current_policy.layer.weight, atol=1.0e-6)


def test_gradient_accumulation_updates_only_after_configured_steps():
    torch.manual_seed(0)
    batch = Batch(torch.randn(4, 2), torch.randn(4))
    policy = BatchPolicy()
    before = policy.layer.weight.detach().clone()
    trainer = DiffusionNFTTrainer(
        current_policy=policy,
        old_policy=copy.deepcopy(policy),
        loss_fn=batch_loss,
        config={"optim": {"lr": 0.05, "grad_clip_norm": 100.0, "microbatch_size": 2, "gradient_accumulation_steps": 2}},
        device="cpu",
    )

    first = trainer.train_step(batch)
    assert first["status"] == "accumulating"
    assert trainer.state.optimizer_step == 0
    assert not trainer.can_checkpoint()
    assert torch.equal(policy.layer.weight, before)

    second = trainer.train_step(batch)
    assert second["status"] == "ok"
    assert trainer.state.optimizer_step == 1
    assert trainer.can_checkpoint()
    assert not torch.equal(policy.layer.weight, before)
