#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$(dirname "$TEST_DIR")"
PROJECT_ROOT="$(dirname "$SCRIPTS_DIR")"
PYTHON="${PYTHON:-python3}"

cd "$SCRIPTS_DIR"
sh ./compile.sh

cd "$TEST_DIR"
bash ./call_tests_cpu.sh
bash ./call_tests_gpu.sh

shopt -s nullglob
par_files=(./*.par)
if ((${#par_files[@]} == 0)); then
    echo "[ERROR] No .par files generated" >&2
    exit 1
fi

echo "[INFO] Verifying ${#par_files[@]} particle files"
verification_failed=0
for par_file in "${par_files[@]}"; do
    if ! "$PYTHON" "$PROJECT_ROOT/py_apps/verify_par_file.py" "$par_file"; then
        verification_failed=1
    fi
done

exit "$verification_failed"
