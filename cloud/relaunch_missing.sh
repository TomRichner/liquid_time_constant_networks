#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# relaunch_missing.sh — Relaunch experiments missing results in GCS
# ─────────────────────────────────────────────────────────────────────
# After a run (especially with spot VMs), scan GCS for missing results
# and relaunch only those jobs. Forces ON-DEMAND VMs so preempted jobs
# complete reliably on the retry pass.
#
# Usage:
#   ./cloud/relaunch_missing.sh <run_name>
#   ./cloud/relaunch_missing.sh <run_name> --models "lstm srnn" --seeds 3
#   ./cloud/relaunch_missing.sh <run_name> --dry-run
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

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [--seeds N] [--models \"m1 m2\"] [--dry-run] [extra flags]"
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

EXPERIMENTS=(har gesture occupancy smnist traffic power ozone-fixed person cheetah)

echo "═══════════════════════════════════════════════════════════════"
echo "  Relaunch Missing — ${RUN_NAME}"
echo "  Experiments:    ${#EXPERIMENTS[@]} (${EXPERIMENTS[*]})"
echo "  Models:         ${#SELECTED_MODELS[@]} (${SELECTED_MODELS[*]})"
echo "  Seeds:          1-${MAX_SEEDS}"
echo "  Override:       ${OVERRIDE:-none}"
echo "  Dry run:        ${DRY_RUN}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Scan for missing results ──────────────────────────────────────────
echo "── Scanning GCS for missing results... ──"
MISSING_JOBS=()
COMPLETE=0
RUNNING=0
TERMINATED_VMS=()

for exp in "${EXPERIMENTS[@]}"; do
    source "${SCRIPT_DIR}/experiments/${exp}.env" 2>/dev/null || continue
    for model in "${SELECTED_MODELS[@]}"; do
        for seed in $(seq 1 ${MAX_SEEDS}); do
            RESULT="${GCS_BUCKET}/results-py/${RUN_NAME}/${model}/${EXPERIMENT_NAME}/seed${seed}/training_log.txt"

            if gcloud storage stat "${RESULT}" &>/dev/null; then
                COMPLETE=$((COMPLETE + 1))
                continue
            fi

            # No result — check if a VM is still running
            VM_NAME=$(echo "${RUN_NAME}-${model}-${EXPERIMENT_NAME}-seed${seed}" | tr '_' '-')
            VM_STATUS=$(gcloud compute instances describe "${VM_NAME}" \
                --zone="${GCP_ZONE}" \
                --format="value(status)" 2>/dev/null || echo "GONE")

            if [ "${VM_STATUS}" = "RUNNING" ]; then
                echo "  ⏳ ${model}/${exp}/seed${seed} — VM still running, skipping"
                RUNNING=$((RUNNING + 1))
                continue
            fi

            if [ "${VM_STATUS}" = "TERMINATED" ] || [ "${VM_STATUS}" = "STOPPED" ]; then
                echo "  🗑  ${model}/${exp}/seed${seed} — stale VM (${VM_STATUS}), will delete"
                TERMINATED_VMS+=("${VM_NAME}")
            fi

            echo "  ✗  ${model}/${exp}/seed${seed} — MISSING"
            MISSING_JOBS+=("${exp}|${model}|${seed}")
        done
    done
done

TOTAL_CHECKED=$(( COMPLETE + RUNNING + ${#MISSING_JOBS[@]} ))
echo ""
echo "  Results:  ${COMPLETE} complete, ${RUNNING} in progress, ${#MISSING_JOBS[@]} missing"
echo "  Total:    ${TOTAL_CHECKED} checked"

if [ ${#MISSING_JOBS[@]} -eq 0 ]; then
    echo ""
    echo "  ✅ Nothing to relaunch!"
    exit 0
fi

# ── Dry run ───────────────────────────────────────────────────────────
if [ "${DRY_RUN}" = true ]; then
    echo ""
    echo "── Dry run — would relaunch ${#MISSING_JOBS[@]} jobs (on-demand): ──"
    for job in "${MISSING_JOBS[@]}"; do
        IFS='|' read -r exp model seed <<< "$job"
        echo "  [dry-run] launch_run.sh ${RUN_NAME} ${exp} ${model} ${seed} ${OVERRIDE}"
    done
    if [ ${#TERMINATED_VMS[@]} -gt 0 ]; then
        echo ""
        echo "  Would delete ${#TERMINATED_VMS[@]} stale VMs:"
        for vm in "${TERMINATED_VMS[@]}"; do
            echo "    gcloud compute instances delete ${vm} --zone=${GCP_ZONE} --quiet"
        done
    fi
    exit 0
fi

# ── Delete stale terminated VMs ───────────────────────────────────────
if [ ${#TERMINATED_VMS[@]} -gt 0 ]; then
    echo ""
    echo "── Deleting ${#TERMINATED_VMS[@]} stale VMs... ──"
    for vm in "${TERMINATED_VMS[@]}"; do
        echo "  Deleting ${vm}..."
        gcloud compute instances delete "${vm}" --zone="${GCP_ZONE}" --quiet 2>/dev/null || true
    done
fi

# ── Force on-demand for relaunches ────────────────────────────────────
export GCP_USE_SPOT=false

echo ""
echo "── Relaunching ${#MISSING_JOBS[@]} missing jobs (ON-DEMAND) ──"

# ── Count currently running VMs ───────────────────────────────────────
get_running_count() {
    gcloud compute instances list \
        --filter="status=RUNNING" \
        --format="value(name)" 2>/dev/null | wc -l | tr -d ' '
}

# ── Launch a single job (background-able) ─────────────────────────────
launch_one() {
    local exp=$1 model=$2 seed=$3
    echo "  → ${exp} / ${model} / seed ${seed}"
    if "${SCRIPT_DIR}/launch_run.sh" "${RUN_NAME}" "${exp}" "${model}" "${seed}" ${OVERRIDE} > /dev/null 2>&1; then
        echo "  ✓ ${exp} / ${model} / seed ${seed}"
    else
        echo "  ✗ FAILED: ${exp} / ${model} / seed ${seed}"
    fi
}

# ── Main launch loop (same batching as launch_all_fast.sh) ────────────
LAUNCHED=0
IDX=0
TOTAL=${#MISSING_JOBS[@]}

CURRENT_RUNNING=$(get_running_count)
echo "  Currently running VMs: ${CURRENT_RUNNING}"
AVAILABLE=$((MAX_CONCURRENT_VMS - CURRENT_RUNNING))

# Phase 1: Burst launch in batches
if [ "${AVAILABLE}" -gt 0 ]; then
    echo ""
    echo "── Phase 1: Burst launching (batches of ${BATCH_SIZE}, ${BATCH_PAUSE}s pause) ──"

    while [ ${IDX} -lt ${TOTAL} ] && [ ${AVAILABLE} -gt 0 ]; do
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

        PIDS=()
        for i in $(seq 1 ${BATCH_COUNT}); do
            IFS='|' read -r exp model seed <<< "${MISSING_JOBS[$IDX]}"
            launch_one "$exp" "$model" "$seed" &
            PIDS+=($!)
            IDX=$((IDX + 1))
            LAUNCHED=$((LAUNCHED + 1))
        done

        for pid in "${PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        AVAILABLE=$((AVAILABLE - BATCH_COUNT))

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
        while true; do
            CURRENT_RUNNING=$(get_running_count)
            if [ "${CURRENT_RUNNING}" -lt "${MAX_CONCURRENT_VMS}" ]; then
                break
            fi
            echo "  $(date +%H:%M:%S) — ${CURRENT_RUNNING} VMs running (limit: ${MAX_CONCURRENT_VMS}), waiting 30s..."
            sleep 30
        done

        IFS='|' read -r exp model seed <<< "${MISSING_JOBS[$IDX]}"
        echo "── (${IDX}/${TOTAL}) Launching ${exp} / ${model} / seed ${seed}..."
        if "${SCRIPT_DIR}/launch_run.sh" "${RUN_NAME}" "${exp}" "${model}" "${seed}" ${OVERRIDE}; then
            LAUNCHED=$((LAUNCHED + 1))
        else
            echo "  ⚠ FAILED: ${exp} / ${model} / seed ${seed}"
            LAUNCHED=$((LAUNCHED + 1))
        fi
        IDX=$((IDX + 1))
        sleep 2
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Relaunch complete: ${LAUNCHED}/${TOTAL} missing jobs submitted"
echo "  All relaunched VMs are ON-DEMAND (no preemption risk)"
echo "  Monitor: ./cloud/monitor.sh ${RUN_NAME}"
echo "═══════════════════════════════════════════════════════════════"
