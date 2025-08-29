#!/bin/bash

cd "$(dirname "$0")/.."

rm -rf build

mkdir build 

cd build 

cmake .. 

cmake --build .

cd src 

./gpu_p_sfc_exe $1 $2