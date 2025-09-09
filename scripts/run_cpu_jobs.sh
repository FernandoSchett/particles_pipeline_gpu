#!/bin/bash

cd "$(dirname "$0")"

CORES_PER_NODE=128 
PARTITION=dc-cpu
TIMES=5

for pp in 3 
do
    for np in 1 2 4 
    do
        NODES=$(( (np + CORES_PER_NODE - 1) / CORES_PER_NODE ))
        echo "Submitting CPU job: np=$np, nodes=$NODES, partition=$PARTITION"
        
        sbatch \
            --nodes=$NODES \
            --ntasks=$np \
            --cpus-per-task=1 \
            --time=00:10:00 \
            --partition=$PARTITION \
            --account=gsp25 \
            --job-name=exp_np${np} \
            --output=exp_cpu${np}_%j.out \
            --error=exp_cpu${np}_%j.err \
            ./run_cpu_experiment.sh $np torus $pp $TIMES
    done
done