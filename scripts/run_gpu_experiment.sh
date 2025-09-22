#!/bin/bash

source ./load_modules_gpu.sh

distribuition=$1
power_particles=$2
times=$3
nprocs=$4 
SEED=$5
mode=$6

echo "Running with echo $CUDA_VISIBLE_DEVICES gpus, $times times"

for i in $(seq 1 $times);
do
    echo "Execution #$i at $distribuition dist and $power_particles power particles $mode mode (SEED: $5)..."
      srun -n $nprocs ../build/src/gpu_mpi_p_sfc_exe $distribuition $power_particles $SEED $mode
done