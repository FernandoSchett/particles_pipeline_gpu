#!/bin/bash

# Load All modules need at the experiment.
module --force purge

module load Stages/2025
module load GCC/13.3.0s

module load UCX/default

module load OpenMPI/5.0.5
module load CUDA
module load UCX-settings/RC-CUDA
module load MPI-settings/CUDA

module load CMake/3.29.3
module load Boost/1.86.0
