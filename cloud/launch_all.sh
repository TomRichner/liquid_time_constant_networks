#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# launch_all.sh — Launch all experiments × all models × all seeds
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/launch_all.sh <run_name>                        # all 8 models, 5 seeds
#   ./cloud/launch_all.sh <run_name> --models "lstm srnn"   # specific models
#   ./cloud/launch_all.sh <run_name> --seeds 3              # seeds 1-3
#   ./cloud/launch_all.sh <run_name> --dry-run              # preview only
#   ./cloud/launch_all.sh <run_name> --epochs 10            # override epochs
#
# Enforces MAX_CONCURRENT_VMS from config.env (default 40).
# Launches in waves, waiting for slots to free up before each wave.
#
# Full run: 9 experiments × 8 models × 5 seeds = 360 VMs
# At 40 concurrent, launches in ~9 waves.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# Defaults
MAX_SEEDS=5
OVERRIDE=""
DRY_RUN=false
SELECTED_MODELS=("${MODELS[@]}")  # all models from config.env

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [--seeds N] [--models \"m1 m2\"] [--dry-run] [extra flags]"
    echo "  Extra flags (--epochs, --size, --lr) are passed to each launch_run.sh call"
    exit 1
fi

RUN_NAME=$1
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --seeds)   MAX_SEEDS=$2; shift 2 ;;
        --models)  IFS=' ' read -ra SELECTED_MODELS <<< "$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --)        shift; OVERRIDE="${OVERRIDE} $*"; break ;;
        *)         OVERRIDE="${OVERRIDE} $1"; shift ;;
    esac
done

EXPERIMENTS=(har gesture occupancy smnist traffic power ozone person cheetah)
TOTAL_VMS=$(( ${#EXPERIMENTS[@]} * ${#SELECTED_MODELS[@]} * MAX_SEEDS ))

echo "═══════════════════════════════════════════════════════════════"
echo "  Full Experiment Launch: ${RUN_NAME}"
echo "  Experiments:    ${#EXPERIMENTS[@]} (${EXPERIMENTS[*]})"
echo "  Models:         ${#SELECTED_MODELS[@]} (${SELECTED_MODELS[*]})"
echo "  Seeds:          1-${MAX_SEEDS}"
echo "  Total VMs:      ${TOTAL_VMS}"
echo "  Max concurrent: ${MAX_CONCURRENT_VMS}"
echo "  Override:       ${OVERRIDE:-none}"
echo "  Dry run:        ${DRY_RUN}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Helper: wait until running VMs < threshold ───────────────────────
wait_for_slots() {
    local max_running=$1
    while true; do
        RUNNING=$(gcloud compute instances list \
            --filter="status=RUNNING" \
            --format="value(name)" 2>/dev/null | wc -l | tr -d ' ')
        if [ "${RUNNING}" -lt "${max_running}" ]; then
            return
        fi
        echo "  $(date +%H:%M:%S) — ${RUNNING} VMs running (limit: ${max_running}), waiting 60s..."
        sleep 60
    done
}

# ── Launch, respecting concurrency limit ─────────────────────────────
LAUNCHED=0
FAILED=0

for exp in "${EXPERIMENTS[@]}"; do
    for model in "${SELECTED_MODELS[@]}"; do
        for seed in $(seq 1 ${MAX_SEEDS}); do
            if [ "${DRY_RUN}" = true ]; then
                echo "  [dry-run] launch_run.sh ${RUN_NAME} ${exp} ${model} ${seed} ${OVERRIDE}"
                continue
            fi

            # Wait for a slot below the concurrency limit
            wait_for_slots "${MAX_CONCURRENT_VMS}"

            echo "── Launching ${exp} / ${model} / seed ${seed}..."
            if "${SCRIPT_DIR}/launch_run.sh" "${RUN_NAME}" "${exp}" "${model}" "${seed}" ${OVERRIDE}; then
                LAUNCHED=$((LAUNCHED + 1))
            else
                echo "  ⚠ FAILED: ${exp} / ${model} / seed ${seed}"
                FAILED=$((FAILED + 1))
            fi
            sleep 2
        done
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
if [ "${DRY_RUN}" = true ]; then
    echo "  Dry run complete. ${TOTAL_VMS} VMs would be launched."
else
    echo "  All launches submitted: ${LAUNCHED} succeeded, ${FAILED} failed"
fi
echo "  Monitor: ./cloud/monitor.sh ${RUN_NAME}"
echo "═══════════════════════════════════════════════════════════════"
