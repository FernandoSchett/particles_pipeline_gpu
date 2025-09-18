#!/bin/bash 

cd "$(dirname "$0")"

GPU_PER_NODE=4 
PARTITION=booster
TIMES=1
SEED=69

LOGDIR="gpu_logdir"

if [ -d "$LOGDIR" ]; then
  rm -rf "$LOGDIR"
fi

mkdir -p "$LOGDIR"

for pp in 9
do
    for ngpu in 1 2 8 
    do
        NODES=$(( (ngpu + GPU_PER_NODE - 1) / GPU_PER_NODE ))
        echo "Submitting GPU job: pp=$pp, np=$ngpu, nodes=$NODES, partition=$PARTITION"
        
        sbatch \
            --nodes=$NODES \
            --ntasks=$ngpu \
            --cpus-per-task=1 \
            --gpus-per-task=1 \
            --time=05:00:00 \
            --partition=$PARTITION \
            --account=gsp25 \
            --job-name=exp_pp${pp}_gpu${ngpu} \
            --output=${LOGDIR}/exp_pp${pp}_gpu${ngpu}_%j.out \
            --error=${LOGDIR}/exp_pp${pp}_gpu${ngpu}_%j.err \
            ./run_gpu_experiment.sh torus $pp $TIMES $ngpu $SEED
    done
done