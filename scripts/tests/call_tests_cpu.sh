#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Config
PARTITION="${PARTITION:-dc-cpu}"
ACCOUNT="${ACCOUNT:-pepcexa}"
TIME="${TIME:-00:20:00}"

rm -f ./*.par core.* || true

MAX_NP="${MAX_NP:-8}"

echo "[INFO] salloc on partition=$PARTITION account=$ACCOUNT time=$TIME (max ntasks=$MAX_NP)"
salloc -p "$PARTITION" -A "$ACCOUNT" -t "$TIME" --ntasks="$MAX_NP" bash -lc "bash ./run_tests_cpu.sh"
