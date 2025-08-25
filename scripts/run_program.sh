#!/bin/bash

cd "$(dirname "$0")/.."

rm -rf build

mkdir build 

cd build 

cmake .. 

cmake --build .

mpirun -np $1 ./p_sfc_exe $2 $3