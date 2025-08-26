# JUN + NANDO COOL MERGING ALGORITH CODE.

## How to run:

1. Clone the repo.
2. Get libs:

         git submodule init
         git submodule update --recursive

3. (OPTIONAL) If in a supercomputer:

         source scripts/load_modules.sh
4. Run:

         sh scripts/run_program.sh <number_processes> <initial_distribution> <power_to_particle_numbers>

5. See Results:

            source scripts/setup_py_env.sh
            python3 visualize.py

## How reproduce the experiments:

1. Run:

         sh scripts/run_jobs.sh 
