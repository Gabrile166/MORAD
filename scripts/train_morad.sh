#!/usr/bin/env bash
# Main MORAD run (Tables 1-3, Figure 3).
#
# Post-trains the pretrained RIDE policy with the MORAD reward on the 527
# training targets: 4 GPUs x (1 target x 8 candidates) per update, three epochs
# = 390 updates, lr 2e-4 with 5% warmup and cosine decay to 0.3x.
#
# Every 10 updates (and before the first one) a policy snapshot is written to
# outputs/morad/validation/policy_step<NNNNNN>.pt and scored on the 153 test
# targets in a background process; the scores are appended to
# outputs/morad/validation/async_results.jsonl (Figure 3). The reported model
# is the snapshot after update 390.
#
#   bash scripts/train_morad.sh
#   bash scripts/train_morad.sh --seed 1          # extra arguments go to train.py
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f scripts/env.sh ]]; then source scripts/env.sh; fi

# The update count and learning-rate schedule assume 4 processes.
NPROC="${NPROC:-4}"

torchrun --standalone --nproc_per_node="${NPROC}" train.py \
    --config configs/morad.yaml --mode train --output-dir outputs/morad "$@"

echo "final policy: outputs/morad/validation/policy_step000390.pt"
echo "evaluate it:  bash scripts/evaluate_test153.sh morad outputs/morad/validation/policy_step000390.pt"
