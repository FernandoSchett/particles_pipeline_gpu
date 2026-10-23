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

        mpirun -np <number_of_processes> ./build/src/p_sfc_exe <distribution_name> <power_particles> <seed> <mode>
        
        mpirun -np <number_of_processes> ./build/src/gpu_mpi_p_sfc_exe <distribution_name> <power_particles> <seed> <mode>
   
6. See Results (Only if power_particles < 4):

            source scripts/setup_py_env.sh
            python3 py_apps/visualise.py
 
## How reproduce the experiments:

1. Run:

         sh scripts/run_all_jobs.sh 

## PEPC input on GPU

From the project root, use the bundled PEPC snapshot or pass another file as the
last argument:

```bash
mpirun -np 1 ./build/src/gpu_mpi_p_sfc_exe pepc_shape 3 67 strong table results/particles/mpi/particle_000000000001.mpi
```

The reader supplies the actual particle count. All MPI ranks share a cubic
Morton domain computed from the global coordinate minima and largest axis span.
Key generation subtracts the domain origin without changing the stored physical
coordinates. Existing PEPC keys are replaced with this pipeline's Morton keys.
The bundled snapshot has 10,000 particles, regardless of the `power` argument;
`power < 4` still enables debug particle and tree output. A fixed input snapshot
does not grow with the process count when `weak` is selected.

Run the domain checks, including the bundled snapshot, with:

```bash
ctest --test-dir build -R p_sfc_particle_domain --output-on-failure
```

## Dependencies:

    - MPI
    - CMake
    - Boost
    - CUDA
    
## Next Steps:

1. Tests creation;
2. Strip Mining for more particles;
3. Tree creation;
4. Comparison with Article Distribution Alg[1] and T-digest statistics Alg;
5. Add more space filling curves;
  
## References

[1] TSUZUKI, S.; AOKI, T. Effective Dynamic Load Balance using Space-Filling Curves for Large-scale SPH Simulations on GPU-rich Supercomputers. In: WORKSHOP ON LATEST ADVANCES IN SCALABLE ALGORITHMS FOR LARGE-SCALE SYSTEMS (ScalA), 7., 2016, Salt Lake City, UT. Proceedings… Piscataway, NJ: IEEE, 2016. p. 1–8. DOI: 10.1109/ScalA.2016.5.
