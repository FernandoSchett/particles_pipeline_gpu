#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Config
PARTITION="${GPU_PARTITION:-dc-gpu}"
ACCOUNT="${ACCOUNT:-pepcexa}"
TIME="${GPU_TIME:-00:20:00}"
MAX_NP="${GPU_MAX_NP:-8}"
GPU_PER_NODE="${GPU_PER_NODE:-4}"
NODES=$(( (MAX_NP + GPU_PER_NODE - 1) / GPU_PER_NODE ))
export GPU_PER_NODE

rm -f ./particle_file_gpu_*.par ./tree_file_gpu_*.gtree core.* || true

echo "[INFO] salloc partition=$PARTITION account=$ACCOUNT time=$TIME GPUs=$MAX_NP nodes=$NODES"
salloc \
  --partition="$PARTITION" \
  --account="$ACCOUNT" \
  --time="$TIME" \
  --nodes="$NODES" \
  --ntasks="$MAX_NP" \
  --ntasks-per-node="$GPU_PER_NODE" \
  --gpus-per-task=1 \
  bash -lc "bash ./run_tests_gpu.sh"
