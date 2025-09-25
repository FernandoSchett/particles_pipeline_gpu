#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Config
PARTITION="${PARTITION:-dc-cpu}"
ACCOUNT="${ACCOUNT:-gsp25}"
TIME="${TIME:-00:20:00}"

# Test matrix
DISTS=("box" "torus")
SEEDS=(69 24)
PP=4
MODE=weak
NP_LIST=(1 2 4 8)

rm -f ./*.par core.* || true

MAX_NP=${NP_LIST[-1]}

echo "[INFO] salloc on partition=$PARTITION account=$ACCOUNT time=$TIME (max ntasks=$MAX_NP)"
salloc -p "$PARTITION" -A "$ACCOUNT" -t "$TIME" --ntasks="$MAX_NP" bash -lc "./run_inside_allocation.sh"
