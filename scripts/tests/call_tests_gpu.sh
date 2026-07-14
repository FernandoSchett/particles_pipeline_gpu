#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Config
PARTITION="${GPU_PARTITION:-booster}"
ACCOUNT="${ACCOUNT:-gsp25}"
TIME="${GPU_TIME:-00:20:00}"
MAX_NP="${GPU_MAX_NP:-8}"

rm -f ./particle_file_gpu_*.par core.* || true

echo "[INFO] salloc on partition=$PARTITION account=$ACCOUNT time=$TIME (max GPUs=$MAX_NP)"
salloc \
  --partition="$PARTITION" \
  --account="$ACCOUNT" \
  --time="$TIME" \
  --ntasks="$MAX_NP" \
  --gpus-per-task=1 \
  bash -lc "bash ./run_tests_gpu.sh"
