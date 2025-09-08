#!/bin/bash

# Load All modules need at the experiment.
module --force purge 
module load Stages/2024 GCC Boost/1.82 OpenMPI CMake CUDA UCX-settings/RC-CUDA MPI-settings/CUDA
module save particles_modules