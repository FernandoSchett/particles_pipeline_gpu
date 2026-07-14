#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")"
source ../load_modules_gpu.sh

DISTS=(box)
SEEDS=(67)
PP="${PP:-3}"
MODE="${MODE:-weak}"
NP_LIST=(1 2 4 8)
GPU_PER_NODE="${GPU_PER_NODE:-4}"

for dist in "${DISTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for np in "${NP_LIST[@]}"; do
      nodes=$(( (np + GPU_PER_NODE - 1) / GPU_PER_NODE ))
      tasks_per_node=$(( np < GPU_PER_NODE ? np : GPU_PER_NODE ))

      echo "[RUN][GPU] dist=$dist pp=$PP seed=$seed mode=$MODE np=$np nodes=$nodes"
      srun \
        --nodes="$nodes" \
        --ntasks="$np" \
        --ntasks-per-node="$tasks_per_node" \
        --gpus-per-task=1 \
        ../../build/src/gpu_mpi_p_sfc_exe "$dist" "$PP" "$seed" "$MODE"

      generated=(particle_file_gpu_n"${np}"_*.par)
      if ((${#generated[@]} != 1)); then
        echo "[ERROR] Expected one GPU .par file for np=$np, found ${#generated[@]}" >&2
        exit 1
      fi

      mv "${generated[0]}" \
        "particle_file_gpu_${dist}_seed${seed}_${MODE}_n${np}_pp${PP}.par"
    done
  done
done
