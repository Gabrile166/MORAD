#!/usr/bin/env bash
# Reward comparison (Table 4).
#
# Trains two policies from the same pretrained RIDE checkpoint with identical
# data, optimizer, schedule and hardware; only the reward differs:
#
#   structural_only   GDT-TS / TM-score / RMSD only        (+ Structural reward)
#   morad_reward      the six-component MORAD reward       (+ MORAD reward)
#
# Each run takes two epochs (263 updates on 4 GPUs). Both are then evaluated at
# the snapshot after update 260 with the standard test protocol.
#
#   bash scripts/train_reward_comparison.sh                  # both arms
#   bash scripts/train_reward_comparison.sh morad_reward     # one arm
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f scripts/env.sh ]]; then source scripts/env.sh; fi

NPROC="${NPROC:-4}"
arms=("$@")
if [[ ${#arms[@]} -eq 0 ]]; then
    arms=(structural_only morad_reward)
fi

for arm in "${arms[@]}"; do
    config="configs/ablation/${arm}.yaml"
    [[ -f "${config}" ]] || { echo "unknown arm: ${arm}" >&2; exit 2; }
    echo "=== training ${arm}"
    torchrun --standalone --nproc_per_node="${NPROC}" train.py \
        --config "${config}" --mode train --output-dir "outputs/ablation/${arm}"
done

for arm in "${arms[@]}"; do
    snapshot="outputs/ablation/${arm}/validation/policy_step000260.pt"
    echo "=== evaluating ${arm} (${snapshot})"
    bash scripts/evaluate_test153.sh "${arm}" "${snapshot}"
done

echo "Table 4: python scripts/eval/collect_tables.py --table4 \\"
echo "    --runs RIDE=ride,\"+ Structural reward\"=structural_only,\"+ MORAD reward\"=morad_reward"
