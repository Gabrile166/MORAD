# MORAD environment. Copy to scripts/env.sh, adjust the paths, then:
#
#   source scripts/env.sh
#
# The defaults assume the workspace layout created by scripts/setup_third_party.sh:
#
#   workspace/
#     MORAD/          this repository
#     third_party/    RhoFold, rhofold_protocol, USalign, EternaFold, RiboDiffusion

_morad_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MORAD_THIRD_PARTY="${MORAD_THIRD_PARTY:-$(cd "${_morad_repo}/.." && pwd)/third_party}"

# RhoFold+ runs in its own environment (PyTorch 2.6 / CUDA 12.4).
export RHOFOLD_PYTHON="${RHOFOLD_PYTHON:-${HOME}/miniconda3/envs/rhofold-plus/bin/python}"
export RHOFOLD_CHECKPOINT="${RHOFOLD_CHECKPOINT:-${MORAD_THIRD_PARTY}/rhofold_protocol/checkpoints/rhofold_pretrained_params.pt}"

# Structure and secondary-structure tools.
export USALIGN_BINARY="${USALIGN_BINARY:-${MORAD_THIRD_PARTY}/USalign/USalign}"
export RNAFOLD_BINARY="${RNAFOLD_BINARY:-$(command -v RNAfold || echo RNAfold)}"
export ETERNAFOLD_BINARY="${ETERNAFOLD_BINARY:-${MORAD_THIRD_PARTY}/EternaFold/src/contrafold}"
export ETERNAFOLD_PARAMS="${ETERNAFOLD_PARAMS:-${MORAD_THIRD_PARTY}/EternaFold/parameters/EternaFoldParams.v1}"

# ViennaRNA's Python bindings from conda-forge may need the environment's
# libstdc++ on older systems:
# export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"

unset _morad_repo
