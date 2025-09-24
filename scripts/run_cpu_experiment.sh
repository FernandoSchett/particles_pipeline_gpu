#!/bin/bash
source ./load_modules_cpu.sh

NP=$1
distribuition=$2
power_particles=$3
times=$4
SEED=$5
mode=$6

echo "Running with $NP processes, $times iterations"

for i in $(seq 1 $times);
do
    #SEED=$((BASE_SEED + i))
    echo "Execution #$i at $distribuition dist with $NP processes and $power_particles power particles $mode mode (SEED: $SEED)..."
    srun -n $NP ../build/src/p_sfc_exe $distribuition $power_particles $SEED $mode
done
