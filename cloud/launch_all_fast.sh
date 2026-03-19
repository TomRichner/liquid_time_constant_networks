#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# launch_all_fast.sh — Fast parallel launcher for all experiments
# ─────────────────────────────────────────────────────────────────────
# Same interface as launch_all.sh, but launches VMs in parallel batches
# of 8 (with 45s pauses) until hitting MAX_CONCURRENT_VMS, then falls
# back to sequential as slots free up.
#
# Usage:
#   ./cloud/launch_all_fast.sh <run_name>
#   ./cloud/launch_all_fast.sh <run_name> --models "lstm srnn" --seeds 1
#   ./cloud/launch_all_fast.sh <run_name> --epochs 200 --dry-run
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

BATCH_SIZE=8
BATCH_PAUSE=15

# Defaults
MAX_SEEDS=5
OVERRIDE=""
DRY_RUN=false
SELECTED_MODELS=("${MODELS[@]}")
SELECTED_EXPERIMENTS=()

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [--seeds N] [--models \"m1 m2\"] [--experiments \"e1 e2\"] [--dry-run] [extra flags]"
    exit 1
fi

RUN_NAME=$1
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --seeds)   MAX_SEEDS=$2; shift 2 ;;
        --models)  IFS=' ' read -ra SELECTED_MODELS <<< "$2"; shift 2 ;;
        --experiments) IFS=' ' read -ra SELECTED_EXPERIMENTS <<< "$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --)        shift; OVERRIDE="${OVERRIDE} $*"; break ;;
        *)         OVERRIDE="${OVERRIDE} $1"; shift ;;
    esac
done

ALL_EXPERIMENTS=(har gesture occupancy smnist traffic power ozone-fixed person cheetah)
if [ ${#SELECTED_EXPERIMENTS[@]} -eq 0 ]; then
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
else
    EXPERIMENTS=("${SELECTED_EXPERIMENTS[@]}")
fi
TOTAL_VMS=$(( ${#EXPERIMENTS[@]} * ${#SELECTED_MODELS[@]} * MAX_SEEDS ))

echo "═══════════════════════════════════════════════════════════════"
echo "  Fast Experiment Launch: ${RUN_NAME}"
echo "  Experiments:    ${#EXPERIMENTS[@]} (${EXPERIMENTS[*]})"
echo "  Models:         ${#SELECTED_MODELS[@]} (${SELECTED_MODELS[*]})"
echo "  Seeds:          1-${MAX_SEEDS}"
echo "  Total VMs:      ${TOTAL_VMS}"
echo "  Max concurrent: ${MAX_CONCURRENT_VMS}"
echo "  Batch size:     ${BATCH_SIZE} (${BATCH_PAUSE}s between batches)"
echo "  Override:       ${OVERRIDE:-none}"
echo "  Dry run:        ${DRY_RUN}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Build the full job list ────────────────────────────────────────────
JOBS=()
for exp in "${EXPERIMENTS[@]}"; do
    for model in "${SELECTED_MODELS[@]}"; do
        for seed in $(seq 1 ${MAX_SEEDS}); do
            JOBS+=("${exp}|${model}|${seed}")
        done
    done
done

if [ "${DRY_RUN}" = true ]; then
    for job in "${JOBS[@]}"; do
        IFS='|' read -r exp model seed <<< "$job"
        echo "  [dry-run] launch_run.sh ${RUN_NAME} ${exp} ${model} ${seed} ${OVERRIDE}"
    done
    echo ""
    echo "  Dry run complete. ${#JOBS[@]} VMs would be launched."
    exit 0
fi

# ── Count currently running VMs ────────────────────────────────────────
get_running_count() {
    gcloud compute instances list \
        --filter="status=RUNNING" \
        --format="value(name)" 2>/dev/null | wc -l | tr -d ' '
}

# ── Launch a single job (background-able) ──────────────────────────────
launch_one() {
    local exp=$1 model=$2 seed=$3
    echo "  → ${exp} / ${model} / seed ${seed}"
    if "${SCRIPT_DIR}/launch_run.sh" "${RUN_NAME}" "${exp}" "${model}" "${seed}" ${OVERRIDE} > /dev/null 2>&1; then
        echo "  ✓ ${exp} / ${model} / seed ${seed}"
    else
        echo "  ✗ FAILED: ${exp} / ${model} / seed ${seed}"
    fi
}

# ── Main launch loop ──────────────────────────────────────────────────
LAUNCHED=0
FAILED=0
IDX=0
TOTAL=${#JOBS[@]}

CURRENT_RUNNING=$(get_running_count)
echo "Currently running VMs: ${CURRENT_RUNNING}"
AVAILABLE=$((MAX_CONCURRENT_VMS - CURRENT_RUNNING))

# Phase 1: Burst launch in batches of BATCH_SIZE while under the limit
if [ "${AVAILABLE}" -gt 0 ]; then
    echo ""
    echo "── Phase 1: Burst launching (batches of ${BATCH_SIZE}, ${BATCH_PAUSE}s pause) ──"

    while [ ${IDX} -lt ${TOTAL} ] && [ ${AVAILABLE} -gt 0 ]; do
        # How many to launch in this batch
        BATCH_COUNT=${BATCH_SIZE}
        if [ ${BATCH_COUNT} -gt ${AVAILABLE} ]; then
            BATCH_COUNT=${AVAILABLE}
        fi
        REMAINING=$((TOTAL - IDX))
        if [ ${BATCH_COUNT} -gt ${REMAINING} ]; then
            BATCH_COUNT=${REMAINING}
        fi

        echo ""
        echo "  Batch: launching ${BATCH_COUNT} VMs (${IDX}/${TOTAL} done, ${AVAILABLE} slots available)"

        # Launch batch in parallel
        PIDS=()
        for i in $(seq 1 ${BATCH_COUNT}); do
            IFS='|' read -r exp model seed <<< "${JOBS[$IDX]}"
            launch_one "$exp" "$model" "$seed" &
            PIDS+=($!)
            IDX=$((IDX + 1))
            LAUNCHED=$((LAUNCHED + 1))
        done

        # Wait for all launches in this batch to complete
        for pid in "${PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        AVAILABLE=$((AVAILABLE - BATCH_COUNT))

        # Pause between batches (unless we're done or out of slots)
        if [ ${IDX} -lt ${TOTAL} ] && [ ${AVAILABLE} -gt 0 ]; then
            echo "  Pausing ${BATCH_PAUSE}s before next batch..."
            sleep ${BATCH_PAUSE}
        fi
    done
fi

# Phase 2: Sequential launch as slots free up
if [ ${IDX} -lt ${TOTAL} ]; then
    echo ""
    echo "── Phase 2: Sequential launching (waiting for slots) ──"

    while [ ${IDX} -lt ${TOTAL} ]; do
        # Wait until a slot opens
        while true; do
            CURRENT_RUNNING=$(get_running_count)
            if [ "${CURRENT_RUNNING}" -lt "${MAX_CONCURRENT_VMS}" ]; then
                break
            fi
            echo "  $(date +%H:%M:%S) — ${CURRENT_RUNNING} VMs running (limit: ${MAX_CONCURRENT_VMS}), waiting 30s..."
            sleep 30
        done

        IFS='|' read -r exp model seed <<< "${JOBS[$IDX]}"
        echo "── (${IDX}/${TOTAL}) Launching ${exp} / ${model} / seed ${seed}..."
        if "${SCRIPT_DIR}/launch_run.sh" "${RUN_NAME}" "${exp}" "${model}" "${seed}" ${OVERRIDE}; then
            LAUNCHED=$((LAUNCHED + 1))
        else
            echo "  ⚠ FAILED: ${exp} / ${model} / seed ${seed}"
            LAUNCHED=$((LAUNCHED + 1))
            FAILED=$((FAILED + 1))
        fi
        IDX=$((IDX + 1))
        sleep 2
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All launches submitted: ${LAUNCHED}/${TOTAL}"
echo "  Monitor: ./cloud/monitor.sh ${RUN_NAME}"
echo "═══════════════════════════════════════════════════════════════"
