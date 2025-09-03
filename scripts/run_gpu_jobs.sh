#!/bin/bash 

GPU_PER_NODE=4 
PARTITION=booster
TIMES=1

for pp in 3
do
    for ngpu in 1 2
    do
        NODES=$(( (ngpu + GPU_PER_NODE - 1) / GPU_PER_NODE ))
        echo "Submitting job: np=$ngpu, nodes=$NODES, partition=$PARTITION"
        
        xenv sbatch \
            --nodes=$NODES \
            --ntasks=1 \
            --cpus-per-task=1 \
            --gpus-per-task=$ngpu \
            --time=00:10:00 \
            --partition=$PARTITION \
            --account=gsp25 \
            --job-name=exp_gpu${ngpu} \
            --output=exp_gpu${ngpu}_%j.out \
            --error=exp_gpu${ngpu}_%j.err \
            ./run_gpu_experiment.sh box $pp $TIMES
    done
done