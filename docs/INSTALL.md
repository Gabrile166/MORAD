# Installation

MORAD runs on Linux with NVIDIA GPUs. The reported run used 4 GPUs for training;
evaluation and the smoke tests run on a single GPU.

## 1. Workspace layout

All paths in the configs are relative to the repository, with external tools in a
sibling `third_party/` directory:

```text
workspace/
├── MORAD/                 # this repository
│   ├── artifacts/         # checkpoints and data pools (scripts/download_assets.sh)
│   └── outputs/           # training runs and evaluations
└── third_party/           # RhoFold, rhofold_protocol, USalign, EternaFold, RiboDiffusion
```

```bash
mkdir workspace && cd workspace
git clone https://github.com/Gabrile166/MORAD.git
cd MORAD
```

System packages: `git`, `curl`, a C++ compiler and `make`
(Ubuntu: `sudo apt-get install -y git curl build-essential`).

## 2. External tools

```bash
bash scripts/setup_third_party.sh
```

The script clones each tool at a pinned commit into `../third_party` and builds
US-align and EternaFold.

| Tool | Role |
|---|---|
| [RhoFold+](https://github.com/ml4bio/RhoFold) | structure prediction for the reward and for evaluation |
| [US-align](https://github.com/pylelab/USalign) | C1′ TM-score |
| [EternaFold](https://github.com/eternagame/EternaFold) | secondary structure for the pairing metrics |
| [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) | MFE, partition function and ensemble defect (installed with conda below) |
| [RiboDiffusion](https://github.com/ml4bio/RiboDiffusion) | baseline sequences |

## 3. Python environments

RIDE and RhoFold+ need different PyTorch/CUDA builds, so they live in two
environments.

```bash
# MORAD: training, sampling, evaluation
conda create -n morad python=3.10 -y
conda activate morad
pip install -r requirements-ride-evaluation.txt
conda install -y -c conda-forge -c bioconda viennarna

# RhoFold+: structure prediction worker
conda create -n rhofold-plus python=3.10 -y
conda activate rhofold-plus
pip install -r requirements-rhofold-plus.txt
```

| Environment | Python | PyTorch |
|---|---|---|
| `morad` | 3.10 | 2.1.2 + CUDA 11.8 |
| `rhofold-plus` | 3.10 | 2.6.0 + CUDA 12.4 |

## 4. Environment variables

```bash
cp scripts/env.example.sh scripts/env.sh   # edit RHOFOLD_PYTHON if needed
source scripts/env.sh
```

| Variable | Default |
|---|---|
| `RHOFOLD_PYTHON` | `~/miniconda3/envs/rhofold-plus/bin/python` |
| `RHOFOLD_CHECKPOINT` | `../third_party/rhofold_protocol/checkpoints/rhofold_pretrained_params.pt` |
| `USALIGN_BINARY` | `../third_party/USalign/USalign` |
| `RNAFOLD_BINARY` | `RNAfold` on `PATH` |
| `ETERNAFOLD_BINARY`, `ETERNAFOLD_PARAMS` | `../third_party/EternaFold/...` |

The run scripts in `scripts/` source `scripts/env.sh` automatically when it exists.

## 5. Checkpoints and data

```bash
bash scripts/download_assets.sh
```

| Component | Destination | Source |
|---|---|---|
| `ride` | `artifacts/models/ride/checkpoint.h5` | [RIDER](https://github.com/COLA-Laboratory/RIDER) release (SHA-256 checked) |
| `rhofold` | `../third_party/rhofold_protocol/checkpoints/rhofold_pretrained_params.pt` | [RhoFold+ weights](https://huggingface.co/cuhkaih/rhofold) (SHA-256 checked) |
| `data` | `artifacts/pools/{train527.pt, test153.pt, test153_native_2d.json}` | [MORAD-RNA/MORAD-Targets](https://huggingface.co/datasets/MORAD-RNA/MORAD-Targets) (SHA-256 checked) |
| `morad` | `artifacts/models/morad/morad_update390.pt` | [MORAD-RNA/MORAD-RIDE](https://huggingface.co/MORAD-RNA/MORAD-RIDE) (SHA-256 checked) |

Single components can be fetched with, e.g., `bash scripts/download_assets.sh ride rhofold`.
See [DATA.md](DATA.md) for the content of each file.

## 6. Check the installation

```bash
conda activate morad
python -m pytest -q                      # unit tests, no GPU or data needed
python train.py --mode smoke             # end-to-end loop with a fake oracle
python verify_setup.py                   # checkpoint, RhoFold+, US-align, RNAfold, pool
```
