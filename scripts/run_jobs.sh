#!/bin/bash

CORES_PER_NODE=128 
PARTITION=dc-cpu

cd "$(dirname "$0")/.."

rm -rf build
rm -f results.csv
mkdir build
cd build

cmake ..
cmake --build .

for pp in 8 
done
    for np in 1 2 4 8 16 32 64 128 256
    do
        NODES=$(( (np + CORES_PER_NODE - 1) / CORES_PER_NODE ))
        echo "Submitting job: np=$np, nodes=$NODES, partition=$PARTITION"
        
        sbatch \
            --nodes=$NODES \
            --ntasks=$np \
            --cpus-per-task=1 \
            --time=00:10:00 \
            --partition=$PARTITION \
            --account=gsp25 \
            --job-name=exp_np${np} \
            --output=exp_np${np}_%j.out \
            --error=exp_np${np}_%j.err \
            ../scripts/run_experiment.sh $np $pp
    done
done