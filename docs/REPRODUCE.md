# Reproducing the paper

Each table and figure in the paper is produced by one command in this repository.
The commands below assume that the installation in [INSTALL.md](INSTALL.md) is
complete, `scripts/env.sh` is configured and the assets are downloaded.

| Paper item | Command | Output |
|---|---|---|
| MORAD training | `bash scripts/train_morad.sh` | `outputs/morad/` |
| Tables 1–3, sequence diversity | `bash scripts/evaluate_test153.sh <model>` → `collect_tables.py` | `outputs/eval/test153/<model>/` |
| Table 4 (reward comparison) | `bash scripts/train_reward_comparison.sh` | `outputs/ablation/<arm>/` |
| Checkpoint comparison | `evaluate.py --config configs/eval/test153_8cand.yaml` → `collect_tables.py --checkpoints` | `outputs/eval/checkpoints/` |
| Figure 3 | `python scripts/plot_training_curve.py` | `outputs/morad/training_curve.png` |

The values reported in the paper are listed in [`results/`](../results).

---

## 1. Train MORAD

```bash
bash scripts/train_morad.sh
```

This launches `train.py` with `configs/morad.yaml` on four GPUs (`torchrun`, one
process per GPU).

| Setting | Value | Config key |
|---|---|---|
| Base policy | pretrained RIDE, 10,214,213 parameters | `data.ride_checkpoint` |
| Training / evaluation targets | 527 / 153 | `data.pool_manifest`, `validation.pool_manifest` |
| Epochs / updates | 3 / 390 | `runtime.epochs`, `runtime.budget.max_outer_steps` |
| Batch per update | 4 targets × 8 candidates (one target per GPU) | `rollout.group_size` |
| Sampler | 50 steps (predict x₀, re-noise), temperature 1 within [0.2, 1.5]; one resampling round at +0.1 | `rollout` |
| Reward | six-component MORAD reward, weighted geometric mean of desirability scores | `reward.composer.preset: morad` |
| Advantage | group-relative z-score, clipped at 2 | `advantage` |
| Objective | DiffusionNFT-style, β = 0.1, λ_ref = 0.02, K_t = 8 | `trainer.loss` |
| Optimizer | AdamW, lr 2×10⁻⁴, weight decay 0, gradient clip 1; 5% warmup, cosine decay to 0.3× | `trainer.optim` |
| EMA sampling policy | ρ_u = min(0.999, (1 + u) / (10 + u)) | `trainer.ema` |
| Held-out evaluation | every 10 updates, update 0 to 390 | `validation` |

Outputs:

```text
outputs/morad/
├── events.jsonl                  # per-update training log
├── checkpoints/                  # latest.pt / final.pt (resumable training state)
└── validation/
    ├── policy_step000000.pt      # policy snapshot at every held-out evaluation
    ├── ...
    ├── policy_step000390.pt      # the reported MORAD model
    └── async_results.jsonl       # held-out scores (Figure 3)
```

To resume an interrupted run:

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/morad.yaml \
    --mode resume --resume outputs/morad/checkpoints/latest.pt --output-dir outputs/morad
```

The reward weights are derived in `scripts/reward/reward_design.py`, and the
reward is implemented in `src/rl/reward.py` (presets `morad` and
`structural_only`).

## 2. Evaluate on the 153 test targets

`scripts/evaluate_test153.sh` runs the full evaluation of one model and writes
every metric reported in Tables 1–3 and the diversity table.

```bash
bash scripts/evaluate_test153.sh ride                       # pretrained RIDE
bash scripts/evaluate_test153.sh morad                      # released MORAD checkpoint
bash scripts/evaluate_test153.sh morad outputs/morad/validation/policy_step000390.pt   # your own run
```

| Step | Script | Output |
|---|---|---|
| 1. Sample 8 designs per target (seed 20260823, temperature 0.8); fold candidate 0 with RhoFold+ and score it | `evaluate.py` | `evaluation.json`, `summary.json`, `candidates.jsonl` |
| 2. Attach the native base pairs | `scripts/eval/inject_native_2d.py` | `evaluation_2d.json` |
| 3. EternaFold pairing, ViennaRNA thermodynamics, sequence diversity | `scripts/eval/run_sstt.py` | `sstt.json` |
| 4. C4′ lDDT, pLDDT and coarse clash | `scripts/eval/fold_all_candidates.py`, `scripts/eval/score_folded.py` | `folded_scores.json` |

Environment variables: `GPUS` (folding GPUs for step 4, e.g. `GPUS=0,1,2,3`),
`RIDE_DEVICE` / `RHOFOLD_DEVICE` (GPU for step 1), `WORKERS` (CPU workers,
default 16), `EVAL_ROOT` (default `outputs/eval/test153`).

### Baselines

The baselines are run with their official code and evaluated with the same
script. Each baseline directory holds eight designs per target as
`fasta/<target_id>_<k>.fasta`, `k = 0…7`, one sequence per file.

**RiboDiffusion** ([ml4bio/RiboDiffusion](https://github.com/ml4bio/RiboDiffusion), cloned by `setup_third_party.sh`):

```bash
python scripts/baselines/pool_to_pdb.py --pool artifacts/pools/test153.pt \
    --out-dir outputs/baselines/ribodiffusion/pdb
python scripts/baselines/run_ribodiffusion_batch.py \
    --python /path/to/ribodiffusion/env/bin/python \
    --repo ../third_party/RiboDiffusion \
    --input-dir outputs/baselines/ribodiffusion/pdb \
    --output-dir outputs/baselines/ribodiffusion --n-samples 8
bash scripts/evaluate_test153.sh --fasta ribodiffusion outputs/baselines/ribodiffusion
```

**gRNAde** ([chaitjo/geometric-rna-design](https://github.com/chaitjo/geometric-rna-design))
at sampling temperatures 0.1 and 0.8, and **RDesign**
([A4Bio/RDesign](https://github.com/A4Bio/RDesign)): generate eight designs per
target from the PDB files written by `pool_to_pdb.py`, store them in the layout
above, then

```bash
bash scripts/evaluate_test153.sh --fasta grnade_t01 outputs/baselines/grnade_t01
bash scripts/evaluate_test153.sh --fasta grnade_t08 outputs/baselines/grnade_t08
bash scripts/evaluate_test153.sh --fasta rdesign    outputs/baselines/rdesign
```

### Collect Tables 1–3 and the diversity table

```bash
python scripts/eval/collect_tables.py --root outputs/eval/test153 \
    --runs RiboDiffusion=ribodiffusion,"gRNAde (T=0.1)"=grnade_t01,"gRNAde (T=0.8)"=grnade_t08,RDesign=rdesign,RIDE=ride,MORAD=morad \
    --out outputs/eval/test153/tables.md
```

| Table | Protocol | Source |
|---|---|---|
| Table 1 (tertiary structure) | candidate 0 of each of the 153 targets | `summary.json`, `folded_scores.json` |
| Table 2 (base pairing) | candidate 0; EternaFold pairs vs. native pairs (137 targets) | `sstt.json` → `paper_single_sample` |
| Table 3 (thermodynamics) | 153 targets | `sstt.json` → `paper_single_sample` |
| Sequence diversity | all 8 candidates of each target | `sstt.json` → `all_candidates` |

## 3. Reward comparison (Table 4)

```bash
bash scripts/train_reward_comparison.sh
```

Two policies are trained from the same pretrained RIDE with the same data,
optimizer, schedule and hardware; only the reward differs.

| Arm | Config | Reward |
|---|---|---|
| `structural_only` (+ Structural reward) | `configs/ablation/structural_only.yaml` | GDT-TS, TM-score and RMSD |
| `morad_reward` (+ MORAD reward) | `configs/ablation/morad_reward.yaml` | six-component MORAD reward |

Each arm runs two epochs (263 updates), and its last policy snapshot
(`outputs/ablation/<arm>/validation/policy_step000260.pt`) is evaluated with
`evaluate_test153.sh`. Together with the RIDE evaluation from Section 2:

```bash
python scripts/eval/collect_tables.py --root outputs/eval/test153 --table4 \
    --runs RIDE=ride,"+ Structural reward"=structural_only,"+ MORAD reward"=morad_reward
```

Structural columns (GDT, TM, RMSD, C1′ TM) use candidate 0; the pairing,
thermodynamic and sequence columns use all eight candidates of each target.

## 4. Checkpoint comparison

Updates 180, 230, 340 and 390 of the MORAD run, with all eight candidates of
each target folded and scored:

```bash
for u in 180 230 340 390; do
    RIDE_CHECKPOINT=outputs/morad/validation/policy_step000${u}.pt \
    python evaluate.py --config configs/eval/test153_8cand.yaml \
        --output-dir outputs/eval/checkpoints/update${u}
done
python scripts/eval/collect_tables.py --root outputs/eval/checkpoints --checkpoints \
    --runs 180=update180,230=update230,340=update340,390=update390
```

Each value is the per-target mean over the eight candidates, then the mean ±
standard error across targets.

## 5. Training curve (Figure 3)

```bash
python scripts/plot_training_curve.py --results outputs/morad/validation/async_results.jsonl
```

This writes the curve of all 40 held-out evaluations (candidate 0 of each target,
updates 0–390) with a centred five-point moving average. The released curve
can be plotted directly:

```bash
python scripts/plot_training_curve.py --csv results/training_curve.csv
```

## Metric reference

| Paper column | JSON key | Tool |
|---|---|---|
| GDT-TS, TM-score (C4′), RMSD | `reward_c4p_gdt_ts`, `reward_c4p_tm_score`, `reward_c4p_rmsd` | RhoFold+, C4′ superposition |
| TM-score (C1′) | `rhofold_c1prime_tm_score` | RhoFold+, US-align |
| Success | `reward_good` | RhoFold+, C4′ superposition |
| Recovery | `sequence_recovery` | — |
| lDDT (C4′), pLDDT, coarse clash | `lddt`, `plddt`, `clash_score` | RhoFold+ |
| Precision, Recall, F1, MCC | `secondary_precision`, `secondary_recall`, `secondary_f1`, `secondary_mcc` | EternaFold vs. native pairs |
| MFE, EFE | `mfe`, `ensemble_free_energy` | ViennaRNA |
| ED, NED (ED/nt) | `ensemble_defect`, `ensemble_defect_per_nt` | ViennaRNA |
| PE (H_pair) | `positional_entropy` | ViennaRNA |
| MFEfreq (p_MFE) | `mfe_frequency` | ViennaRNA |
| GC | `gc_content` | — |
| Hamming, 3-mer diversity | `hamming_diversity`, `kmer_diversity_3` | — |
