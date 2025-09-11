#!/bin/bash

cd "$(dirname "$0")"

source ./load_modules_cpu.sh

CORES_PER_NODE=48 
PARTITION=batch
TIMES=5

LOGDIR="cpu_logdir"
mkdir -p "$LOGDIR"

for pp in 8 9 
do
    for np in 1 2 4 6 8 12 16 32 50 64 128 256
    do
        NODES=$(( (np + CORES_PER_NODE - 1) / CORES_PER_NODE ))
        echo "Submitting CPU job: pp${pp}, np=$np, nodes=$NODES, partition=$PARTITION"
        
        sbatch \
            --nodes=$NODES \
            --ntasks=$np \
            --cpus-per-task=1 \
            --time=05:00:00 \
            --partition=$PARTITION \
            --account=gsp25 \
            --job-name=exp_pp${pp}_np${np} \
            --output=${LOGDIR}/exp_pp${pp}_cpu${np}_%j.out \
            --error=${LOGDIR}/exp_pp${pp}_cpu${np}_%j.err \
            ./run_cpu_experiment.sh $np torus $pp $TIMES
    done
done