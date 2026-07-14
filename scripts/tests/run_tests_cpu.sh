#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")"

DISTS=(box)
SEEDS=(67)
PP="${PP:-4}"
MODE="${MODE:-weak}"
NP_LIST=(1 2 4 128 256)

for dist in "${DISTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for np in "${NP_LIST[@]}"; do
      echo "[RUN] dist=$dist pp=$PP seed=$seed mode=$MODE np=$np"
      srun -n "$np" ../../build/src/p_sfc_exe "$dist" "$PP" "$seed" "$MODE"

      generated=(particle_file_cpu_n"${np}"_*.par)
      if ((${#generated[@]} != 1)); then
        echo "[ERROR] Expected one .par file for np=$np, found ${#generated[@]}" >&2
        exit 1
      fi
      mv "${generated[0]}" "particle_file_cpu_${dist}_seed${seed}_${MODE}_n${np}_pp${PP}.par"
    done
  done
done
