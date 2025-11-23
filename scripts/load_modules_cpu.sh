#!/bin/bash

# Load All modules need at the experiment.
module --force purge

module load Stages/2025
module load GCC/13.3.0

module load Boost/1.86.0
module load OpenMPI/5.0.5

module load CMake/3.29.3

module load CUDA