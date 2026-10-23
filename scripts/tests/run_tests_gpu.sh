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
source ../load_modules_gpu.sh

DISTS=(pepc_shape)
SEEDS=(67)
PP="${PP:-3}"
MODE="${MODE:-strong}"
[[ "$PP" =~ ^[0-3]$ ]] || { echo "PP must be between 0 and 3 to write debug files" >&2; exit 1; }
NP_LIST=(1 2 4 8)
GPU_PER_NODE="${GPU_PER_NODE:-4}"
MAX_NP="${GPU_MAX_NP:-8}"

rm -f "$P_SFC_OUTPUT_DIR"/particle_file_gpu_*.par "$P_SFC_OUTPUT_DIR"/tree_file_gpu_*.gtree

for dist in "${DISTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for np in "${NP_LIST[@]}"; do
      ((np <= MAX_NP)) || continue
      nodes=$(( (np + GPU_PER_NODE - 1) / GPU_PER_NODE ))
      tasks_per_node=$(( np < GPU_PER_NODE ? np : GPU_PER_NODE ))

      echo "[RUN][GPU] dist=$dist pp=$PP seed=$seed mode=$MODE np=$np nodes=$nodes"
      srun \
        --nodes="$nodes" \
        --ntasks="$np" \
        --ntasks-per-node="$tasks_per_node" \
        --gpus-per-task=1 \
        "$PROJECT_ROOT/build/src/gpu_mpi_p_sfc_exe" "$dist" "$PP" "$seed" "$MODE" table "$INPUT_FILE"

      generated=("$P_SFC_OUTPUT_DIR"/particle_file_gpu_n"${np}"_*.par)
      if ((${#generated[@]} != 1)); then
        echo "[ERROR] Expected one GPU .par file for np=$np, found ${#generated[@]}" >&2
        exit 1
      fi

      mv "${generated[0]}" \
        "$P_SFC_OUTPUT_DIR/particle_file_gpu_${dist}_seed${seed}_${MODE}_n${np}_pp${PP}.par"

      generated_trees=("$P_SFC_OUTPUT_DIR"/tree_file_gpu_n"${np}"_*.gtree)
      if ((${#generated_trees[@]} != 1)); then
        echo "[ERROR] Expected one GPU .gtree file for np=$np, found ${#generated_trees[@]}" >&2
        exit 1
      fi
      mv "${generated_trees[0]}" \
        "$P_SFC_OUTPUT_DIR/tree_file_gpu_${dist}_seed${seed}_${MODE}_n${np}_pp${PP}.gtree"
    done
  done
done
