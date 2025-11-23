#!/bin/bash 

cd "$(dirname "$0")"

GPU_PER_NODE=4 
PARTITION=dc-gpu-devel 
TIMES=1
SEED=69


for mode in weak
do
    LOGDIR="gpu_logdir_s${SEED}_M${mode}"
    
    if [ -d "$LOGDIR" ]; then
      rm -rf "$LOGDIR"
    fi
    
    mkdir -p "$LOGDIR"

    rm  *.par || true
    rm  core.* || true
    
    for pp in 3
    do
        for ngpu in 1 2
        do
            NODES=$(( (ngpu + GPU_PER_NODE - 1) / GPU_PER_NODE ))
            echo "JobName=exp_pp${pp}_gpu${ngpu}_seed${SEED}_${mode}, Mode=$mode, pp=$pp, ngpu=$ngpu, nodes=$NODES, partition=$PARTITION, time:05:00:00"
            
            sbatch \
                --nodes=$NODES \
                --ntasks=$ngpu \
                --cpus-per-task=1 \
                --gpus-per-task=1 \
                --time=00:10:00 \
                --partition=$PARTITION \
                --account=pepcexa \
                --job-name=exp_pp${pp}_gpu${ngpu}_seed${SEED}_M${mode} \
                --output=${LOGDIR}/exp_pp${pp}_gpu${ngpu}_S${SEED}_M${mode}%j.out \
                --error=${LOGDIR}/exp_pp${pp}_gpu${ngpu}_S${SEED}_M${mode}%j.err \
                ./run_gpu_experiment.sh torus $pp $TIMES $ngpu $SEED $mode
        done
    done
done