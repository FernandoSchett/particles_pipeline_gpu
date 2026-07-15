#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$(dirname "$TEST_DIR")"
PROJECT_ROOT="$(dirname "$SCRIPTS_DIR")"
PYTHON="${PYTHON:-python3}"

cd "$PROJECT_ROOT"
rm -rf build
mkdir -p build

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

tree_files=(./*.tree)
if ((${#tree_files[@]} == 0)); then
    echo "[ERROR] No CPU .tree files generated" >&2
    verification_failed=1
else
    echo "[INFO] Verifying ${#tree_files[@]} CPU tree files"
    for tree_file in "${tree_files[@]}"; do
        if ! "$PYTHON" "$PROJECT_ROOT/py_apps/verify_tree_file.py" "$tree_file"; then
            verification_failed=1
        fi
    done
fi

global_tree_files=(./*.gtree)
if ((${#global_tree_files[@]} == 0)); then
    echo "[ERROR] No GPU .gtree files generated" >&2
    verification_failed=1
else
    echo "[INFO] Verifying ${#global_tree_files[@]} GPU global tree files"
    for tree_file in "${global_tree_files[@]}"; do
        if ! "$PYTHON" "$PROJECT_ROOT/py_apps/verify_global_tree_file.py" "$tree_file"; then
            verification_failed=1
        fi
    done
fi

exit "$verification_failed"
