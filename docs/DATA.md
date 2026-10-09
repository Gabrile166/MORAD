# Data and checkpoints

All files below are downloaded by `scripts/download_assets.sh` into `artifacts/`.
The pools are hosted at [Gabriel166/MORAD-data](https://huggingface.co/datasets/Gabriel166/MORAD-data)
and the MORAD checkpoint at [Gabriel166/MORAD](https://huggingface.co/Gabriel166/MORAD).

## Target pools

| File | Targets | Length (nt) | Used for |
|---|---:|---|---|
| `artifacts/pools/train527.pt` | 527 | 27–258 (median 75) | MORAD training |
| `artifacts/pools/test153.pt` | 153 | 15–186 (median 65) | held-out evaluation during training and all reported tables |
| `artifacts/pools/test153_native_2d.json` | 137 | — | native base pairs for the pairing metrics |

The evaluation set is built separately from the training set. It comprises 71
[RNA3DB](https://github.com/marcellszi/rna3db) entries, drawn from RNA3DB
components that contain no training target, and 82 entries curated from the
[RCSB PDB](https://www.rcsb.org/). No evaluation sequence exactly matches a
training sequence. When the native sequences are folded with RhoFold+,
the median GDT-TS is 0.6400 on the training set and 0.6014 on the evaluation set.

### Pool format

Each `.pt` file is a `torch.save`d dictionary with a `targets` list. Every target holds

| Field | Content |
|---|---|
| `metadata` | `target_id`, native `sequence`, `length`, `structure_hash` and source identifiers |
| `ref_backbone_coords` | reference backbone coordinates (P, C4′, N1/N9), `[L, 3, 3]` |
| `ref_c4p_coords` | reference C4′ coordinates, `[L, 3]` |
| `mask_coords` | coordinate-validity mask, `[L]` |
| `raw_record` | backbone features consumed by RIDE and the reference C1′ trace |

`src/rl/targets.py` loads the pools for training and evaluation.

### Native base pairs

`test153_native_2d.json` maps `target_id` to a dot-bracket string of the base
pairs extracted from the native 3D structure. `scripts/eval/inject_native_2d.py`
attaches them to an evaluation run before the pairing metrics are computed; the
137 targets with native pairs form the population of the pairing table.

## Checkpoints

| File | Content |
|---|---|
| `artifacts/models/ride/checkpoint.h5` | pretrained RIDE policy from [RIDER](https://github.com/COLA-Laboratory/RIDER) (revision `799f9a0`), the starting point of every run |
| `artifacts/models/morad/morad_update390.pt` | MORAD policy after 390 updates, `{"model": state_dict}` |

The MORAD file has the same format as the policy snapshots written during
training (`outputs/<run>/validation/policy_step<NNNNNN>.pt`), so any snapshot can
be evaluated the same way:

```bash
RIDE_CHECKPOINT=artifacts/models/morad/morad_update390.pt python evaluate.py --config configs/eval/test153.yaml
```
