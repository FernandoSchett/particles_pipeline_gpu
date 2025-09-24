# JUN + NANDO COOL MERGING ALGORITHM CODE.

## How to run:

1. Clone the repo.
2. Get libs:

         git submodule init
         git submodule update --recursive

3. (OPTIONAL) If in a supercomputer:

        source scripts/load_modules_cpu.sh

        source scripts/load_modules_gpu.sh

4. Compile:

         sh scripts/compile.sh

5.  Run:

        mpirun -np <number_of_processes> ./build./src/p_sfc_exe <distribution_name> <power_particles> <seed> <mode>
        
        ./build./src/gpu_mpi_p_sfc_exe <distribution_name> <power_particles>
   
6. See Results (Only if power_particles < 4):

            source scripts/setup_py_env.sh
            python3 visualize.py
 
## How reproduce the experiments:

1. Run:

         sh scripts/run_all_jobs.sh 

## Dependencies:

    - MPI
    - CMake
    - Boost
    - CUDA