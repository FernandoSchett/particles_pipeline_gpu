#!/bin/bash
# submit_jobs.sh

CORES_PER_NODE=128 
PARTITION=dc-cpu*

for np in 1 2 4 8 16 32 64 128 256
do
    NODES=$(( (np + CORES_PER_NODE - 1) / CORES_PER_NODE ))
    echo "Submitting job: np=$np, nodes=$NODES, partition=$PARTITION"
    
    sbatch --nodes=$NODES --ntasks=$np run_experiment.sh $np
done
