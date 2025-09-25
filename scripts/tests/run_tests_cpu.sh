#!/usr/bin/env bash

cd "$(dirname "$0")"

for dist in "${DISTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for np in "${NP_LIST[@]}"; do
      echo "[RUN] dist=$dist pp=$PP seed=$seed mode=$MODE np=$np"
      srun -n "$np" ../../build/src/p_sfc_exe "$dist" "$PP" "$seed" "$MODE"
    done
  done
done
