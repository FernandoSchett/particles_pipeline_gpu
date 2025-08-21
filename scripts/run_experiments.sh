#!/bin/bash

cd "$(dirname "$0")/.."

rm -rf build

mkdir build 

cd build 

cmake .. 

cmake --build .

for np in 4 8 16 32 64 128
do
    for i in 1 2 3 4 5
    do
        echo "Executing with $np processes..."
        mpirun -np $np ./p_sfc_exe 8
    done
done
