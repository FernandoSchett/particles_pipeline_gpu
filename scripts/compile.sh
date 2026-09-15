#!/bin/sh
set -e

cd "$(dirname "$0")/.."
. ./.env
export CUDACXX

rm -rf build
rm -f results.csv
mkdir build
cd build

cmake .. -DCMAKE_CUDA_COMPILER="$CUDACXX"
cmake --build .
