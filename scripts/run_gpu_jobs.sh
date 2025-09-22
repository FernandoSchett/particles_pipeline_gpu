#!/bin/bash 

cd "$(dirname "$0")"

GPU_PER_NODE=4 
PARTITION=booster
TIMES=5
SEED=69

LOGDIR="gpu_logdir_s$SEED"

if [ -d "$LOGDIR" ]; then
  rm -rf "$LOGDIR"
fi

mkdir -p "$LOGDIR"

for mode in weak strong
do
    LOGDIR="gpu_logdir_s${SEED}_m${mode}"
    
    if [ -d "$LOGDIR" ]; then
      rm -rf "$LOGDIR"
    fi
    
    mkdir -p "$LOGDIR"

    for pp in 8
    do
        for ngpu in 1 2 4 6 8 16 24 32 50 64 128 256
        do
            NODES=$(( (ngpu + GPU_PER_NODE - 1) / GPU_PER_NODE ))
            echo "JobName=exp_pp${pp}_gpu${ngpu}_seed${SEED}_${mode}, Mode=$mode, pp=$pp, ngpu=$ngpu, nodes=$NODES, partition=$PARTITION, time:05:00:00"
            
            sbatch \
                --nodes=$NODES \
                --ntasks=$ngpu \
                --cpus-per-task=1 \
                --gpus-per-task=1 \
                --time=05:00:00 \
                --partition=$PARTITION \
                --account=gsp25 \
                --job-name=exp_pp${pp}_gpu${ngpu}_seed${SEED}_mode${mode} \
                --output=${LOGDIR}/exp_pp${pp}_gpu${ngpu}_seed${SEED}_m${mode}%j.out \
                --error=${LOGDIR}/exp_pp${pp}_gpu${ngpu}_seed${SEED}_m${mode}%j.err \
                ./run_gpu_experiment.sh torus $pp $TIMES $ngpu $SEED $mode
        done
    done
done