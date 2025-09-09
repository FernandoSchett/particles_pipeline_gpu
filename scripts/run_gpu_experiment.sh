#!/bin/bash
cd "$(dirname "$0")"

source ./load_modules.sh

distribuition=$1
power_particles=$2
times=$3
nprocs=$4 

echo "Running with echo $CUDA_VISIBLE_DEVICES gpus, $times times"

for i in $(seq 1 $times);
do
    echo "Execution #$i at $distribuition dist and $power_particles power particles..."
      srun -n $nprocs ../build/src/gpu_mpi_p_sfc_exe $distribution $power_particles
done