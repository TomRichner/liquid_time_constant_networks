#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# build_image.sh — Create a custom VM image with Python + TensorFlow
# ─────────────────────────────────────────────────────────────────────
# Usage:
#   ./cloud/build_image.sh
#
# Creates a reusable disk image so each experiment VM boots ready to
# train in ~30 seconds instead of spending 5+ min on pip install.
#
# Steps:
#   1. Create a temporary VM from Debian base
#   2. SSH in and install Python 3.12, TensorFlow, numpy, pandas
#   3. Create disk image from the VM
#   4. Delete temporary VM
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

IMAGE_NAME="srnn-python-v1"
IMAGE_FAMILY="srnn-python"
TEMP_VM="python-image-builder"
BASE_IMAGE_FAMILY="debian-12"
BASE_IMAGE_PROJECT="debian-cloud"

echo "═══════════════════════════════════════════════════════════════"
echo "  Building Python VM Image"
echo "  Image:  ${IMAGE_NAME} (family: ${IMAGE_FAMILY})"
echo "  Zone:   ${GCP_ZONE}"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Create temporary VM ────────────────────────────────────
echo ""
echo "=== Step 1: Creating temporary VM ==="
gcloud compute instances create "${TEMP_VM}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type=n2-highmem-2 \
    --image-family="${BASE_IMAGE_FAMILY}" \
    --image-project="${BASE_IMAGE_PROJECT}" \
    --boot-disk-size=30GB \
    --no-address \
    --scopes=storage-full,compute-rw

echo "  Waiting 30s for VM to boot..."
sleep 30

# ── Step 2: Install Python + packages ──────────────────────────────
echo ""
echo "=== Step 2: Installing Python + TensorFlow ==="
echo "  SSH into the VM and run the following commands:"
echo ""
echo "  gcloud compute ssh ${TEMP_VM} --zone=${GCP_ZONE} --tunnel-through-iap"
echo ""
echo "  Then inside the VM:"
echo "  ──────────────────────────────────────────────────"
cat << 'INSTALL_EOF'
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip git

  sudo python3 -m venv /opt/python-venv
  sudo /opt/python-venv/bin/pip install tensorflow numpy pandas tqdm

  # Verify
  /opt/python-venv/bin/python3 -c "import tensorflow as tf; print(f'TF {tf.__version__}')"
  /opt/python-venv/bin/python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"

  # Exit SSH
  exit
INSTALL_EOF
echo "  ──────────────────────────────────────────────────"
echo ""
echo "  After SSH installation is complete, press Enter to continue..."
read -p "  > "

# ── Step 3: Stop VM and create image ──────────────────────────────
echo ""
echo "=== Step 3: Stopping VM and creating image ==="
gcloud compute instances stop "${TEMP_VM}" --zone="${GCP_ZONE}" --quiet
echo "  VM stopped."

gcloud compute images create "${IMAGE_NAME}" \
    --source-disk="${TEMP_VM}" \
    --source-disk-zone="${GCP_ZONE}" \
    --family="${IMAGE_FAMILY}" \
    --description="Python 3.12 + TensorFlow for LTC experiments"

echo "  Image '${IMAGE_NAME}' created."

# ── Step 4: Delete temporary VM ────────────────────────────────────
echo ""
echo "=== Step 4: Deleting temporary VM ==="
gcloud compute instances delete "${TEMP_VM}" --zone="${GCP_ZONE}" --quiet
echo "  Temporary VM deleted."

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done! Image '${IMAGE_NAME}' is ready."
echo "  Verify: gcloud compute images describe ${IMAGE_NAME}"
echo "═══════════════════════════════════════════════════════════════"
