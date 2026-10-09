#!/usr/bin/env python
"""Load the upstream RiboDiffusion runtime once and sample many PDB targets."""

from __future__ import annotations

import argparse
import functools
import gc
import importlib.util
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.n_samples < 1:
        parser.error("--n-samples must be positive")

    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import torch
    import tree
    from datasets import utils as data_utils
    from diffusion import NoiseScheduleVP
    from models import ExponentialMovingAverage, create_model
    import run_lib
    from sampling import get_sampling_fn
    from utils import get_data_inverse_scaler, restore_checkpoint

    config = _load_config(repo / "configs" / "inference_ribodiffusion.py")
    config.eval.n_samples = int(args.n_samples)
    run_lib.set_random_seed(config)

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = run_lib.get_optimizer(config, model.parameters())
    state = restore_checkpoint(
        str(repo / "ckpts" / "exp_inf.pth"),
        {"optimizer": optimizer, "model": model, "ema": ema, "step": 0},
        device=config.device,
    )
    state["ema"].copy_to(model.parameters())
    scheduler = NoiseScheduleVP(
        config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1,
    )
    sample = get_sampling_fn(
        config,
        scheduler,
        config.eval.sampling_steps,
        get_data_inverse_scaler(config),
    )
    pdb_to_data = functools.partial(
        data_utils.PDBtoData,
        num_posenc=config.data.num_posenc,
        num_rbf=config.data.num_rbf,
        knn_num=config.data.knn_num,
    )
    fasta_dir = args.output_dir.resolve() / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    for index, pdb_path in enumerate(args.input, start=1):
        path = pdb_path.resolve()
        # Resetting the official seed per target makes resumed runs independent
        # of how many earlier targets were already present.
        run_lib.set_random_seed(config)
        structure = pdb_to_data(str(path))
        batched = tree.map_structure(
            lambda value: value.unsqueeze(0).repeat_interleave(args.n_samples, dim=0).to(config.device),
            structure,
        )
        with torch.no_grad():
            samples = sample(model, batched)
        for sample_index, sequence in enumerate(samples):
            data_utils.sample_to_fasta(
                sequence,
                path.stem,
                str(fasta_dir / f"{path.stem}_{sample_index}.fasta"),
            )
        recovery = samples.eq(batched["seq"]).float().mean().item()
        print(
            f"[{index}/{len(args.input)}] completed {path.stem}; recovery={recovery:.4f}",
            flush=True,
        )
        del structure, batched, samples
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_config(path: Path):
    spec = importlib.util.spec_from_file_location("ribodiffusion_inference_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load RiboDiffusion config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


if __name__ == "__main__":
    main()
