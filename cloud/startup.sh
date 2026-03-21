#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# startup.sh — VM startup script for Python (TF1) experiment runs
# ─────────────────────────────────────────────────────────────────────
# Runs on VM boot (as root). Reads config from VM metadata, downloads
# the dataset, runs training, uploads results to GCS, and self-deletes.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Read VM metadata ─────────────────────────────────────────────────
META_URL="http://metadata.google.internal/computeMetadata/v1/instance"
META_HEADER="Metadata-Flavor: Google"

get_meta() { curl -s "${META_URL}/attributes/$1" -H "${META_HEADER}"; }

EXPERIMENT=$(get_meta experiment)
MODEL=$(get_meta model)
SEED=$(get_meta seed)
TRAIN_ARGS=$(get_meta train-args)
GCS_BUCKET=$(get_meta gcs-bucket)
RUN_NAME=$(get_meta run-name)
GITHUB_REPO=$(get_meta github-repo)
VM_NAME=$(curl -s "${META_URL}/name" -H "${META_HEADER}")
VM_ZONE=$(curl -s "${META_URL}/zone" -H "${META_HEADER}" | awk -F/ '{print $NF}')

RESULT_PATH="${GCS_BUCKET}/results-py/${RUN_NAME}/${MODEL}/${EXPERIMENT}/seed${SEED}"
LOG_FILE="/tmp/training.log"
REPO_DIR="/opt/ltc-repo"

# ── Redirect all output to log file ────────────────────────────────
touch "$LOG_FILE" && chmod 666 "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "═══════════════════════════════════════════════════════════════"
echo "  Python LTC Experiment Runner"
echo "═══════════════════════════════════════════════════════════════"
echo "  VM:         ${VM_NAME}"
echo "  Run:        ${RUN_NAME}"
echo "  Experiment: ${EXPERIMENT}"
echo "  Model:      ${MODEL}"
echo "  Seed:       ${SEED}"
echo "  Args:       ${TRAIN_ARGS}"
echo "  GCS Bucket: ${GCS_BUCKET}"
echo "  Results:    ${RESULT_PATH}"
echo "  Started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "═══════════════════════════════════════════════════════════════"

# ── Error handler: upload log + self-delete on failure ─────────────
cleanup_on_error() {
    local exit_code=$?
    echo ""
    echo "!!! FATAL ERROR (exit code: ${exit_code}) at $(date -u +%Y-%m-%dT%H:%M:%SZ) !!!"
    # Try to upload the error log to GCS
    gcloud storage cp "${LOG_FILE}" "${RESULT_PATH}/ERROR_training_log.txt" 2>/dev/null || true
    # Create error metadata
    cat > /tmp/run_metadata.json <<ERREOF
{
    "run_name": "${RUN_NAME}",
    "experiment": "${EXPERIMENT}",
    "model": "${MODEL}",
    "seed": ${SEED},
    "train_args": "${TRAIN_ARGS}",
    "exit_code": ${exit_code},
    "error": true,
    "vm_name": "${VM_NAME}",
    "failed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ERREOF
    gcloud storage cp /tmp/run_metadata.json "${RESULT_PATH}/" 2>/dev/null || true
    echo "  Error log uploaded to ${RESULT_PATH}/"
    # Self-delete
    sleep 5
    gcloud compute instances delete "${VM_NAME}" --zone="${VM_ZONE}" --quiet 2>/dev/null || true
}
trap cleanup_on_error ERR

# ── Step 1: Clone/update code ──────────────────────────────────────
echo ""
echo "=== Step 1: Getting latest code ==="
if [ -d "${REPO_DIR}" ]; then
    cd "${REPO_DIR}"
    git pull --ff-only || echo "WARNING: git pull failed, using existing version"
else
    # Retry git clone up to 3 times (NAT connectivity can be flaky)
    for attempt in 1 2 3; do
        echo "  Clone attempt ${attempt}/3..."
        if git clone "${GITHUB_REPO}" "${REPO_DIR}"; then
            break
        fi
        if [ $attempt -eq 3 ]; then
            echo "FATAL: git clone failed after 3 attempts"
            exit 1
        fi
        echo "  Retrying in 30s..."
        sleep 30
    done
    cd "${REPO_DIR}"
fi
echo "  Commit: $(git rev-parse --short HEAD)"

# ── Step 2: Download dataset from GCS ──────────────────────────────
echo ""
DATASET_NAME="${EXPERIMENT%_fixed}"  # strip _fixed suffix for data dir
echo "=== Step 2: Downloading dataset '${DATASET_NAME}' ==="
DATASET_DIR="${REPO_DIR}/experiments_with_ltcs/data/${DATASET_NAME}"
mkdir -p "${DATASET_DIR}"

if [ "${DATASET_NAME}" != "smnist" ]; then
    gcloud storage cp -r "${GCS_BUCKET}/datasets/${DATASET_NAME}/*" "${DATASET_DIR}/" 2>&1
    echo "  Downloaded to ${DATASET_DIR}"
    ls -lh "${DATASET_DIR}/"
else
    echo "  SMnist uses Keras download — skipping GCS dataset"
fi

# ── Step 3: Set up Python environment ──────────────────────────────
echo ""
echo "=== Step 3: Setting up Python environment ==="
export TF_USE_LEGACY_KERAS=1
if [ -d "/opt/python-venv" ]; then
    source /opt/python-venv/bin/activate
    echo "  Using pre-built Python venv"
else
    echo "  No pre-built venv found, installing dependencies..."
    python3 -m venv /tmp/python-venv
    source /tmp/python-venv/bin/activate
    pip install --quiet "tensorflow==2.15.*" tf_keras numpy pandas tqdm
    echo "  Installed dependencies"
fi
# Ensure tf_keras is installed (needed for TF_USE_LEGACY_KERAS=1)
python3 -c "import tf_keras" 2>/dev/null || pip install --quiet tf_keras
python3 --version
echo "  TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>&1)"

# ── Step 4: Run training ───────────────────────────────────────────
echo ""
echo "=== Step 4: Starting training ==="
TRAIN_CMD="python3 ${EXPERIMENT}.py --model ${MODEL} --seed ${SEED} ${TRAIN_ARGS}"
echo "  Command: ${TRAIN_CMD}"
echo ""

# ── Compute periodic upload interval ──────────────────────────────
EPOCHS=$(echo "${TRAIN_ARGS}" | sed -n 's/.*--epochs[[:space:]]\+\([0-9]\+\).*/\1/p')
EPOCHS=${EPOCHS:-200}
if [ "${EPOCHS}" -gt 50 ]; then
    UPLOAD_INTERVAL=$(( EPOCHS / 10 ))
else
    UPLOAD_INTERVAL=5
fi
echo "  Periodic GCS upload every ${UPLOAD_INTERVAL} epochs (${EPOCHS} total)"

CKPT_DIR="${REPO_DIR}/experiments_with_ltcs/tf_sessions/${EXPERIMENT}"

# ── Background epoch watcher for periodic GCS uploads ─────────────
# Monitors training log for epoch completions and uploads at intervals.
# Provides progress visibility and spot instance recoverability.
(
    LAST_UPLOADED=0
    while true; do
        sleep 30
        # Parse latest epoch number from training log
        CURRENT_EPOCH=$(grep 'Epochs ' "${LOG_FILE}" 2>/dev/null | tail -1 | sed 's/Epochs \([0-9]*\).*/\1/')
        [ -z "${CURRENT_EPOCH}" ] && continue

        # Check if we've crossed an upload threshold
        NEXT_UPLOAD=$(( LAST_UPLOADED + UPLOAD_INTERVAL ))
        if [ "${CURRENT_EPOCH}" -ge "${NEXT_UPLOAD}" ]; then
            echo "  [periodic-upload] Epoch ${CURRENT_EPOCH}: uploading to GCS..."
            # Upload training log
            gcloud storage cp "${LOG_FILE}" "${RESULT_PATH}/training_log.txt" 2>/dev/null || true
            # Upload checkpoints (all: init, periodic, best, latest)
            if [ -d "${CKPT_DIR}" ]; then
                gcloud storage cp "${CKPT_DIR}/"*.* "${RESULT_PATH}/checkpoint/" 2>/dev/null || true
                [ -f "${CKPT_DIR}/checkpoint" ] && \
                    gcloud storage cp "${CKPT_DIR}/checkpoint" "${RESULT_PATH}/checkpoint/" 2>/dev/null || true
            fi
            # Upload result CSV if it exists yet
            RESULT_DIR="${REPO_DIR}/experiments_with_ltcs/results/${EXPERIMENT}"
            if [ -d "${RESULT_DIR}" ]; then
                gcloud storage cp "${RESULT_DIR}/"*.csv "${RESULT_PATH}/" 2>/dev/null || true
            fi
            # Write progress marker
            echo "{\"epoch\": ${CURRENT_EPOCH}, \"total_epochs\": ${EPOCHS}, \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > /tmp/progress.json
            gcloud storage cp /tmp/progress.json "${RESULT_PATH}/" 2>/dev/null || true
            LAST_UPLOADED=${CURRENT_EPOCH}
            echo "  [periodic-upload] Upload complete at epoch ${CURRENT_EPOCH}"
        fi
    done
) &
WATCHER_PID=$!
echo "  Background upload watcher started (PID: ${WATCHER_PID})"

TRAIN_START=$(date +%s)

cd "${REPO_DIR}/experiments_with_ltcs"
${TRAIN_CMD}

TRAIN_EXIT=$?
TRAIN_END=$(date +%s)
TRAIN_DURATION=$(( TRAIN_END - TRAIN_START ))

# Stop the background watcher
kill ${WATCHER_PID} 2>/dev/null || true
wait ${WATCHER_PID} 2>/dev/null || true

echo ""
echo "  Training exit code: ${TRAIN_EXIT}"
echo "  Training duration:  ${TRAIN_DURATION}s ($(( TRAIN_DURATION / 60 ))m $(( TRAIN_DURATION % 60 ))s)"

# ── Step 5: Upload results to GCS ──────────────────────────────────
echo ""
echo "=== Step 5: Uploading results ==="

# Upload result CSVs
RESULT_DIR="${REPO_DIR}/experiments_with_ltcs/results/${EXPERIMENT}"
if [ -d "${RESULT_DIR}" ]; then
    gcloud storage cp "${RESULT_DIR}"/*.csv "${RESULT_PATH}/"
    echo "  Uploaded result CSVs"
else
    echo "  WARNING: No result directory found at ${RESULT_DIR}"
fi

# Upload model checkpoints (init, periodic, best, last)
# TF Saver saves as file prefix: <model>.index, <model>.data-00000-of-00001
# Named checkpoints use suffixes: _init, _epoch10, _epoch20, ..., _last
CKPT_DIR="${REPO_DIR}/experiments_with_ltcs/tf_sessions/${EXPERIMENT}"
if [ -d "${CKPT_DIR}" ]; then
    gcloud storage cp "${CKPT_DIR}/"*.* "${RESULT_PATH}/checkpoint/"
    # Also upload the TF checkpoint metadata file if it exists
    [ -f "${CKPT_DIR}/checkpoint" ] && gcloud storage cp "${CKPT_DIR}/checkpoint" "${RESULT_PATH}/checkpoint/"
    echo "  Uploaded all checkpoints"
else
    echo "  WARNING: No checkpoint directory at ${CKPT_DIR}"
fi

# Upload training log
gcloud storage cp "${LOG_FILE}" "${RESULT_PATH}/training_log.txt"

# Create and upload run metadata
cat > /tmp/run_metadata.json << EOF
{
    "run_name": "${RUN_NAME}",
    "experiment": "${EXPERIMENT}",
    "model": "${MODEL}",
    "seed": ${SEED},
    "train_args": "${TRAIN_ARGS}",
    "exit_code": ${TRAIN_EXIT},
    "duration_seconds": ${TRAIN_DURATION},
    "vm_name": "${VM_NAME}",
    "completed": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "commit": "$(cd ${REPO_DIR} && git rev-parse --short HEAD)"
}
EOF
gcloud storage cp /tmp/run_metadata.json "${RESULT_PATH}/"
echo "  Uploaded run_metadata.json"

# Final log upload (includes upload messages)
gcloud storage cp "${LOG_FILE}" "${RESULT_PATH}/training_log.txt"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results at:  ${RESULT_PATH}"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 6: Self-delete the VM ─────────────────────────────────────
echo ""
echo "=== Step 6: Self-deleting VM ==="
sleep 5
gcloud compute instances delete "${VM_NAME}" --zone="${VM_ZONE}" --quiet
