#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# launch_run.sh — Launch a single Python experiment VM
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/launch_run.sh <run_name> <experiment> <model> <seed> [--epochs N]
#
# Examples:
#   ./cloud/launch_run.sh prod har lstm 1
#   ./cloud/launch_run.sh smoke10 har srnn 1 --epochs 10
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <run_name> <experiment> <model> <seed> [--epochs N] [--size N] [--lr N]"
    echo "  run_name:   required label (e.g. 'prod', 'smoke10')"
    echo "  experiment: har, gesture, occupancy, smnist, traffic, power, ozone, person, cheetah"
    echo "  model:      lstm, ltc, ltc_rk, ltc_ex, ctrnn, ctgru, node, srnn"
    echo "  seed:       1-5"
    exit 1
fi

RUN_NAME=$1
EXPERIMENT=$2
MODEL=$3
SEED=$4
shift 4

# Parse optional overrides
OVERRIDE_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --epochs)     OVERRIDE_ARGS="${OVERRIDE_ARGS} --epochs $2"; shift 2 ;;
        --size)       OVERRIDE_ARGS="${OVERRIDE_ARGS} --size $2"; shift 2 ;;
        --lr)         OVERRIDE_ARGS="${OVERRIDE_ARGS} --lr $2"; shift 2 ;;
        --batch_size) OVERRIDE_ARGS="${OVERRIDE_ARGS} --batch_size $2"; shift 2 ;;
        *)            OVERRIDE_ARGS="${OVERRIDE_ARGS} $1"; shift ;;
    esac
done

# ── Load config ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

EXP_ENV="${SCRIPT_DIR}/experiments/${EXPERIMENT}.env"
if [ ! -f "${EXP_ENV}" ]; then
    echo "ERROR: Experiment config not found: ${EXP_ENV}"
    echo "Available experiments:"
    ls "${SCRIPT_DIR}/experiments/"*.env | xargs -I{} basename {} .env
    exit 1
fi
source "${EXP_ENV}"

# Apply CLI overrides (appended last → last-wins in argparse)
if [ -n "${OVERRIDE_ARGS}" ]; then
    ARGS="${ARGS}${OVERRIDE_ARGS}"
fi

# ── VM name and machine type ───────────────────────────────────────
VM_NAME=$(echo "${RUN_NAME}-${MODEL}-${EXPERIMENT_NAME}-seed${SEED}" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
VM_MACHINE="${GCP_VM_FAMILY}-${MACHINE_TIER:-${GCP_DEFAULT_TIER}}"

# ── Check concurrency limit ───────────────────────────────────────
echo "=== Pre-launch Check ==="
CURRENT_VMS=$(gcloud compute instances list --filter="status=RUNNING" --format="value(name)" 2>/dev/null | wc -l | tr -d ' ')
echo "  Running VMs:  ${CURRENT_VMS} / ${MAX_CONCURRENT_VMS}"

if [ "${CURRENT_VMS}" -ge "${MAX_CONCURRENT_VMS}" ]; then
    echo "ERROR: Already at max concurrent VMs (${MAX_CONCURRENT_VMS})."
    echo "  Wait for running VMs to finish or increase MAX_CONCURRENT_VMS in config.env."
    exit 1
fi
echo ""

# ── Check if VM already exists ─────────────────────────────────────
if gcloud compute instances describe "${VM_NAME}" --zone="${GCP_ZONE}" &>/dev/null; then
    echo "ERROR: VM '${VM_NAME}' already exists!"
    echo "  To delete: gcloud compute instances delete ${VM_NAME} --zone=${GCP_ZONE} --quiet"
    exit 1
fi

# ── Check if results already exist ─────────────────────────────────
RESULT_CHECK="${GCS_BUCKET}/results-py/${RUN_NAME}/${MODEL}/${EXPERIMENT_NAME}/seed${SEED}/training_log.txt"
if gcloud storage stat "${RESULT_CHECK}" &>/dev/null; then
    echo "WARNING: Results already exist for ${RUN_NAME}/${MODEL}/${EXPERIMENT_NAME}/seed${SEED}"
    read -p "  Overwrite? (y/N): " CONFIRM
    if [ "${CONFIRM}" != "y" ]; then
        echo "  Skipping."
        exit 0
    fi
fi

# ── Create the VM ──────────────────────────────────────────────────
echo "=== Launching VM ==="
echo "  Name:     ${VM_NAME}"
echo "  Machine:  ${VM_MACHINE}"
echo "  Zone:     ${GCP_ZONE}"
echo "  Image:    ${GCP_IMAGE_FAMILY}"
echo "  Model:    ${MODEL}"
echo "  Args:     ${ARGS}"
echo "  Seed:     ${SEED}"
echo ""

SPOT_FLAGS=""
if [ "${GCP_USE_SPOT}" = "true" ]; then
    SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
fi

gcloud compute instances create "${VM_NAME}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type="${VM_MACHINE}" \
    --image-family="${GCP_IMAGE_FAMILY}" \
    --boot-disk-size=15GB \
    --no-address \
    ${SPOT_FLAGS} \
    --metadata="run-name=${RUN_NAME},experiment=${EXPERIMENT_NAME},model=${MODEL},seed=${SEED},train-args=${ARGS},gcs-bucket=${GCS_BUCKET},github-repo=${GITHUB_REPO}" \
    --metadata-from-file=startup-script="${SCRIPT_DIR}/startup.sh" \
    --scopes=storage-full,compute-rw

echo ""
echo "=== VM '${VM_NAME}' launched ==="
echo "  Monitor:  gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} --tunnel-through-iap --command='tail -f /tmp/training.log'"
echo "  Serial:   gcloud compute instances get-serial-port-output ${VM_NAME} --zone=${GCP_ZONE} | tail -20"
echo "  Delete:   gcloud compute instances delete ${VM_NAME} --zone=${GCP_ZONE} --quiet"
