import copy
import random

import numpy as np
import pytest
import torch

from src.rl.checkpoint import CheckpointCursor, _normalize_rng_tensor, load_checkpoint, save_checkpoint


def test_checkpoint_restores_model_optimizer_and_rejects_config_mismatch(tmp_path):
    model = torch.nn.Linear(2, 1)
    old = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.ones(1, 2)
    model(x).sum().backward()
    opt.step()
    path = tmp_path / "ckpt.pt"
    config = {"trainer": {"loss": {"nft_beta": 0.1}}}
    save_checkpoint(
        path,
        current_policy=model,
        old_policy=old,
        reference_policy=None,
        optimizer=opt,
        cursor=CheckpointCursor(outer_step=3, optimizer_step=2, target_cursor=1, oracle_calls=4),
        resolved_config=config,
        reward_scale_state={"std": 2.0},
        temperature_state={"temperature": 0.8},
    )
    restored = torch.nn.Linear(2, 1)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
    payload = load_checkpoint(path, current_policy=restored, optimizer=restored_opt, expected_config=config)
    assert payload["cursor"]["outer_step"] == 3
    assert payload["reward_scale_state"]["std"] == 2.0
    assert torch.allclose(model.weight, restored.weight)
    try:
        load_checkpoint(path, expected_config={"different": True})
    except ValueError as exc:
        assert "config hash mismatch" in str(exc)
    else:
        raise AssertionError("expected mismatch")


def test_checkpoint_restores_rng(tmp_path):
    model = torch.nn.Linear(1, 1)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    save_checkpoint(
        tmp_path / "rng.pt",
        current_policy=model,
        old_policy=None,
        reference_policy=None,
        optimizer=None,
        resolved_config={},
    )
    expected_py = random.random()
    expected_np = np.random.rand()
    expected_torch = torch.rand(1)
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    load_checkpoint(tmp_path / "rng.pt", expected_config={})
    assert random.random() == expected_py
    assert np.random.rand() == expected_np
    assert torch.allclose(torch.rand(1), expected_torch)


def test_rng_tensor_normalization_returns_cpu_uint8_contiguous_tensor():
    source = torch.arange(8, dtype=torch.int64)

    normalized = _normalize_rng_tensor(source)

    assert normalized.device.type == "cpu"
    assert normalized.dtype == torch.uint8
    assert normalized.is_contiguous()
    assert normalized.tolist() == list(range(8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for map_location=cuda RNG regression")
def test_checkpoint_load_with_cuda_map_location_restores_cpu_rng_state(tmp_path):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "cuda_map.pt"
    torch.manual_seed(321)
    save_checkpoint(
        path,
        current_policy=model,
        old_policy=None,
        reference_policy=None,
        optimizer=None,
        resolved_config={},
    )
    expected = torch.rand(1)
    torch.manual_seed(999)

    load_checkpoint(path, expected_config={}, map_location=torch.device("cuda"))

    assert torch.allclose(torch.rand(1), expected)
