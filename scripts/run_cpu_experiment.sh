#!/bin/bash
source ./load_modules_cpu.sh

NP=$1
distribuition=$2
power_particles=$3
times=$4

echo "Running with $NP processes, $times iterations"

for i in $(seq 1 $times);
do
    echo "Execution #$i at $distribuition dist with $NP processes and $power_particles power particles..."
    srun -n $NP ../build/src/p_sfc_exe $distribuition $power_particles
done
