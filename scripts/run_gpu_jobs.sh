#!/bin/bash 

cd "$(dirname "$0")"

GPU_PER_NODE=4 
PARTITION=dc-gpu
TIMES=5

# 1 2 4 8 16 32 64 128 256
# 5 6 8 9 12 16 32 64
for pp in 3
do
    for ngpu in 2 4 
    do
        NODES=$(( (ngpu + GPU_PER_NODE - 1) / GPU_PER_NODE ))
        echo "Submitting GPU job: np=$ngpu, nodes=$NODES, partition=$PARTITION"
        
        sbatch \
            --nodes=$NODES \
            --ntasks=$ngpu \
            --cpus-per-task=1 \
            --gpus-per-task=1 \
            --time=00:10:00 \
            --partition=$PARTITION \
            --account=gsp25 \
            --job-name=exp_gpu${ngpu} \
            --output=exp_gpu${ngpu}_%j.out \
            --error=exp_gpu${ngpu}_%j.err \
            ./run_gpu_experiment.sh box $pp $TIMES $ngpu
    done
done