#!/bin/bash
# Smoke test: run 2 epochs of each model type on each experiment
# Usage: cd experiments_with_ltcs && bash smoke_test.sh

set -o pipefail

MODELS="lstm ctrnn ltc ltc_rk ltc_ex node ctgru srnn"
EXPERIMENTS="har gesture occupancy traffic ozone power person smnist cheetah"

PASS=0
FAIL=0
RESULTS=""

for exp in $EXPERIMENTS; do
  for model in $MODELS; do
    echo ""
    echo "=============================="
    echo "=== $exp / $model ==="
    echo "=============================="
    
    # Run with 2 epochs, log every epoch, small hidden size
    uv run python3 "$exp.py" --model "$model" --epochs 2 --log 1 --size 32 2>&1
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
      STATUS="PASS"
      PASS=$((PASS + 1))
    else
      STATUS="FAIL"
      FAIL=$((FAIL + 1))
    fi
    
    RESULTS="${RESULTS}${exp}\t${model}\t${STATUS}\n"
    echo ">>> $exp / $model: $STATUS (exit $EXIT_CODE)"
  done
done

echo ""
echo "=============================="
echo "=== SMOKE TEST SUMMARY ==="
echo "=============================="
echo -e "Experiment\tModel\tStatus"
echo -e "$RESULTS"
echo "Total: $((PASS + FAIL)) | Pass: $PASS | Fail: $FAIL"
