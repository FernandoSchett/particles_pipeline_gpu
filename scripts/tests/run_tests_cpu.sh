#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$TEST_DIR/../.." && pwd)"
cd "$TEST_DIR"
export P_SFC_OUTPUT_DIR="${P_SFC_OUTPUT_DIR:-$PROJECT_ROOT/results/tests}"
INPUT_FILE="${INPUT_FILE:-$PROJECT_ROOT/results/particles/mpi/particle_000000000001.mpi}"
INPUT_FILE="$(realpath "$INPUT_FILE")"
[[ -r "$INPUT_FILE" ]] || { echo "Cannot read input: $INPUT_FILE" >&2; exit 1; }
mkdir -p "$P_SFC_OUTPUT_DIR"
P_SFC_OUTPUT_DIR="$(cd "$P_SFC_OUTPUT_DIR" && pwd)"
source ../load_modules_cpu.sh

DISTS=(pepc_shape)
SEEDS=(67)
PP="${PP:-3}"
MODE="${MODE:-strong}"
[[ "$PP" =~ ^[0-3]$ ]] || { echo "PP must be between 0 and 3 to write debug files" >&2; exit 1; }
NP_LIST=(1 2 4 128 256)
CPU_PER_NODE="${CPU_PER_NODE:-128}"
MAX_NP="${CPU_MAX_NP:-256}"

rm -f "$P_SFC_OUTPUT_DIR"/particle_file_cpu_*.par "$P_SFC_OUTPUT_DIR"/tree_file_cpu_*.tree

for dist in "${DISTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for np in "${NP_LIST[@]}"; do
      ((np <= MAX_NP)) || continue
      nodes=$(( (np + CPU_PER_NODE - 1) / CPU_PER_NODE ))
      tasks_per_node=$(( np < CPU_PER_NODE ? np : CPU_PER_NODE ))

      echo "[RUN][CPU] dist=$dist pp=$PP seed=$seed mode=$MODE np=$np nodes=$nodes"
      srun \
        --nodes="$nodes" \
        --ntasks="$np" \
        --ntasks-per-node="$tasks_per_node" \
        --cpus-per-task=1 \
        "$PROJECT_ROOT/build/src/p_sfc_exe" "$dist" "$PP" "$seed" "$MODE" table "$INPUT_FILE"

      generated=("$P_SFC_OUTPUT_DIR"/particle_file_cpu_n"${np}"_*.par)
      if ((${#generated[@]} != 1)); then
        echo "[ERROR] Expected one .par file for np=$np, found ${#generated[@]}" >&2
        exit 1
      fi
      mv "${generated[0]}" "$P_SFC_OUTPUT_DIR/particle_file_cpu_${dist}_seed${seed}_${MODE}_n${np}_pp${PP}.par"

      generated_trees=("$P_SFC_OUTPUT_DIR"/tree_file_cpu_n"${np}"_*.tree)
      if ((${#generated_trees[@]} != 1)); then
        echo "[ERROR] Expected one CPU .tree file for np=$np, found ${#generated_trees[@]}" >&2
        exit 1
      fi
      mv "${generated_trees[0]}" \
        "$P_SFC_OUTPUT_DIR/tree_file_cpu_${dist}_seed${seed}_${MODE}_n${np}_pp${PP}.tree"
    done
  done
done

# Exercise small adversarial inputs with the real binary in the same allocation.
if [[ "${RUN_EDGE_CASES:-1}" == 1 ]]; then
    edge_np=$(( MAX_NP < 4 ? MAX_NP : 4 ))
    edge_nodes=$(( (edge_np + CPU_PER_NODE - 1) / CPU_PER_NODE ))
    edge_tasks=$(( edge_np < CPU_PER_NODE ? edge_np : CPU_PER_NODE ))
    "${PYTHON:-python3}" "$PROJECT_ROOT/py_apps/run_tree_correctness.py" \
        --device cpu --executable "$PROJECT_ROOT/build/src/p_sfc_exe" \
        --ranks "$edge_np" --launcher srun --rank-flag=-n \
        --launcher-arg="--nodes=$edge_nodes" --launcher-arg="--ntasks-per-node=$edge_tasks" \
        --launcher-arg=--cpus-per-task=1 \
        --output-dir "$P_SFC_OUTPUT_DIR/edge_cases"
fi
