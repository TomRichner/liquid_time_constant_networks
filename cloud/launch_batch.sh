#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# launch_batch.sh — Launch all seeds for one experiment × one model
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/launch_batch.sh <run_name> <experiment> <model>
#
# Examples:
#   ./cloud/launch_batch.sh prod har lstm        # launches 5 VMs
#   ./cloud/launch_batch.sh prod har srnn
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <run_name> <experiment> <model>"
    echo "  model: lstm, ltc, ltc_rk, ltc_ex, ctrnn, ctgru, node, srnn"
    exit 1
fi

RUN_NAME=$1
EXPERIMENT=$2
MODEL=$3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/experiments/${EXPERIMENT}.env"

N=${N_SEEDS:-5}

echo "=== Batch Launch: ${RUN_NAME} / ${MODEL} / ${EXPERIMENT} (${N} seeds) ==="
echo ""

LAUNCHED=0
SKIPPED=0

for SEED in $(seq 1 ${N}); do
    echo "--- Seed ${SEED}/${N} ---"
    if "${SCRIPT_DIR}/launch_run.sh" "${RUN_NAME}" "${EXPERIMENT}" "${MODEL}" "${SEED}"; then
        LAUNCHED=$((LAUNCHED + 1))
    else
        SKIPPED=$((SKIPPED + 1))
    fi
    echo ""
    sleep 3
done

echo "=== Batch complete: ${LAUNCHED} launched, ${SKIPPED} skipped ==="
