#!/bin/bash
source ./load_modules.sh

distribuition=$1
power_particles=$2
times=$3

echo "Running with echo $CUDA_VISIBLE_DEVICES gpus, $times times"

for i in $(seq 1 $times);
do
    echo "Execution #$i at $distribuition dist with $NP processes and $power_particles power particles..."
    srun -n $NP ../build/src/gpu_p_sfc_exe $distribuition $power_particles
done