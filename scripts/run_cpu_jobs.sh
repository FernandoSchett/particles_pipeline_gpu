#!/bin/bash

cd "$(dirname "$0")"

source ./load_modules_cpu.sh

CORES_PER_NODE=48 
PARTITION=dc-cpu
TIMES=5
SEED=69


for mode in weak strong 
do
    LOGDIR="cpu_logdir_s${SEED}_M${mode}"
    
    if [ -d "$LOGDIR" ]; then
      rm -rf "$LOGDIR"
    fi
    
    mkdir -p "$LOGDIR"

    rm  *.par
    
    for pp in 8
    do
        for np in 1 2 4 6 8 16 24 32 50 64 128 256
        do
            NODES=$(( (np + CORES_PER_NODE - 1) / CORES_PER_NODE ))
            echo "JobName=exp_pp${pp}_cpu${np}_seed${SEED}_${mode}, Mode=$mode, pp=$pp, np=$np, nodes=$NODES, partition=$PARTITION, time=05:00:00"   
            
            sbatch \
                --nodes=$NODES \
                --ntasks=$np \
                --cpus-per-task=1 \
                --time=05:00:00 \
                --partition=$PARTITION \
                --account=gsp25 \
                --job-name=exp_pp${pp}_np${np}_seed${SEED}_M${mode} \
                --output=${LOGDIR}/exp_pp${pp}_cpu${np}_S${SEED}_M${mode}%j.out \
                --error=${LOGDIR}/exp_pp${pp}_cpu${np}_S${SEED}_M${mode}%j.err \
                ./run_cpu_experiment.sh $np torus $pp $TIMES $SEED $mode 
        done
    done
done
    
