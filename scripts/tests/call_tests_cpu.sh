#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Config
PARTITION="${PARTITION:-dc-cpu}"
ACCOUNT="${ACCOUNT:-pepcexa}"
TIME="${TIME:-00:20:00}"
MAX_NP="${CPU_MAX_NP:-256}"
CPU_PER_NODE="${CPU_PER_NODE:-128}"
NODES=$(( (MAX_NP + CPU_PER_NODE - 1) / CPU_PER_NODE ))
TASKS_PER_NODE=$(( MAX_NP < CPU_PER_NODE ? MAX_NP : CPU_PER_NODE ))
export CPU_PER_NODE

rm -f ./*.par ./*.tree ./tree_file_cpu_*.png core.* || true

echo "[INFO] salloc partition=$PARTITION account=$ACCOUNT time=$TIME CPUs=$MAX_NP nodes=$NODES"
salloc \
  --partition="$PARTITION" \
  --account="$ACCOUNT" \
  --time="$TIME" \
  --nodes="$NODES" \
  --ntasks="$MAX_NP" \
  --ntasks-per-node="$TASKS_PER_NODE" \
  --cpus-per-task=1 \
  bash -lc "bash ./run_tests_cpu.sh"
