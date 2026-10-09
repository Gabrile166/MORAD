#!/usr/bin/env bash
# Full evaluation of one model on the 153 test targets (Tables 1-4, diversity).
#
#   bash scripts/evaluate_test153.sh ride
#   bash scripts/evaluate_test153.sh morad    [policy.pt]   # default artifacts/models/morad/morad_update390.pt
#   bash scripts/evaluate_test153.sh NAME     policy.pt     # any MORAD snapshot, e.g. a reward-comparison run
#   bash scripts/evaluate_test153.sh --fasta NAME DIR       # baseline sequences in DIR/fasta/<target_id>_<k>.fasta
#
# Steps (outputs in outputs/eval/test153/NAME/):
#   1. evaluate.py           8 designs per target; RhoFold+ folding and scoring of candidate 0
#                            -> evaluation.json, summary.json
#   2. inject_native_2d.py   native base pairs for the pairing metrics -> evaluation_2d.json
#   3. run_sstt.py           EternaFold pairing, ViennaRNA thermodynamics, diversity -> sstt.json
#   4. fold_all_candidates.py + score_folded.py
#                            C4' lDDT, pLDDT, coarse clash -> folded_scores.json
#
# Environment: source scripts/env.sh first. GPUS selects the folding GPUs for
# step 4 (default 0); RIDE_DEVICE / RHOFOLD_DEVICE select the GPU for step 1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f scripts/env.sh ]]; then source scripts/env.sh; fi

ROOT="${EVAL_ROOT:-outputs/eval/test153}"
GPUS="${GPUS:-0}"
WORKERS="${WORKERS:-16}"
NATIVE_2D="${NATIVE_2D:-artifacts/pools/test153_native_2d.json}"

if [[ "${1:-}" == "--fasta" ]]; then
    [[ $# -eq 3 ]] || { echo "usage: $0 --fasta NAME DIR" >&2; exit 2; }
    name="$2"
    export BASELINE_NAME="${name}" BASELINE_DIR="$(cd "$3" && pwd)"
    config=configs/eval/test153_precomputed.yaml
else
    [[ $# -ge 1 ]] || { sed -n '2,7p' "$0" >&2; exit 2; }
    name="$1"
    config=configs/eval/test153.yaml
    case "${name}" in
        ride) unset RIDE_CHECKPOINT ;;
        morad) export RIDE_CHECKPOINT="${2:-artifacts/models/morad/morad_update390.pt}" ;;
        *) [[ $# -eq 2 ]] || { echo "usage: $0 ${name} policy.pt" >&2; exit 2; }
           export RIDE_CHECKPOINT="$2" ;;
    esac
fi
out="${ROOT}/${name}"
mkdir -p "${out}"

echo "[1/4] evaluate ${name} -> ${out}"
python evaluate.py --config "${config}" --output-dir "${out}"

echo "[2/4] native base pairs"
python scripts/eval/inject_native_2d.py --native-2d "${NATIVE_2D}" --runs "${out}"

echo "[3/4] pairing, thermodynamics and diversity"
python scripts/eval/run_sstt.py --runs "${name}=${out}/evaluation_2d.json" \
    --out "${out}/sstt.json" --workers "${WORKERS}" --secondary-tool eternafold

echo "[4/4] lDDT, pLDDT and coarse clash"
python scripts/eval/fold_all_candidates.py --runs "${name}=${out}" \
    --cache "${ROOT}/fold_cache.json" --scratch "${ROOT}/fold_scratch" --gpus "${GPUS}"
python scripts/eval/score_folded.py --runs "${name}=${out}" \
    --cache "${ROOT}/fold_cache.json" --out "${out}/folded_scores.json" --workers "${WORKERS}"

echo "done: ${out}"
echo "tables: python scripts/eval/collect_tables.py --root ${ROOT} --runs <Label>=<name>,..."
