#!/bin/bash
source ./load_modules.sh

NP=$1
 
echo "Running with $NP processes"

for i in 1 2 3 4 5
do
    echo "Execution #$i at $2 dist with $NP processes and $3 power particles..."
    srun -n $NP ../build/src/p_sfc_exe $2 $3
done
