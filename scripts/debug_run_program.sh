#!/bin/bash

cd "$(dirname "$0")/.."

rm -rf build

mkdir build 

cd build 

cmake .. -DCMAKE_BUILD_TYPE=Debug

cmake --build .

mpirun -np 4 valgrind --leak-check=full --track-origins=yes ./p_sfc_exe