#!/bin/bash
source ./load_modules.sh

NP=$1
 
echo "Running with $NP processes"

for i in 1 2 3 4 5
do
    echo "Execution #$i with $NP processes and $2 power particles..."
    srun -n $NP ../build/p_sfc_exe $2
done
