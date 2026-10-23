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

## PEPC input on CPU and GPU

From the project root, use the bundled PEPC snapshot or pass another file as the
last argument:

```bash
mpirun -np 1 ./build/src/gpu_mpi_p_sfc_exe pepc_shape 3 67 strong table results/particles/mpi/particle_000000000001.mpi
```

Use `./build/src/p_sfc_exe` with the same arguments for the CPU pipeline.

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

## Test inputs and generated outputs

Both Slurm test runners use `pepc_shape` and the bundled 10,000-particle snapshot.
The default mode is `strong`, since the input file has a fixed particle count.
Run from the project root with the cluster modules, Python dependencies, account,
and CUDA compiler path in `.env` configured:

```bash
bash scripts/tests/run_all_tests.sh
```

Set `INPUT_FILE` to an absolute path to test another PEPC snapshot:

```bash
INPUT_FILE=/path/to/particles.mpi bash scripts/tests/run_all_tests.sh
```

Particle and tree files (`.par`, `.tree`, `.gtree`) and timing CSVs are written to:

- `results/debug/` for direct executable runs from the project root.
- `results/tests/` for the Slurm test runners.

Set `P_SFC_OUTPUT_DIR` to override the output directory. The executables create
it automatically. The test runners replace previous particle and tree outputs
for their respective device in this directory, while CSV rows are appended.
`PP` must be between 0 and 3 for these tests to produce the required debug files;
the actual particle count comes from the PEPC file. `CPU_MAX_NP` and `GPU_MAX_NP`
limit both the Slurm reservation and the process counts selected from the test lists.

`visualise.py` also searches both output folders. `gen_results.py` reads timing
CSVs from `results/debug/` by default; use `--csv-dir PATH` for another folder
containing both `results_strong.csv` and `results_weak.csv`.

The PEPC snapshot contains repeated Morton keys. The test suite uses
`verify_par_file.py --key-partitions RANKS` to validate the exact quantile
partition and current owner of every particle, keeping equal keys together.
It does not require an unrealistically uniform particle count across ranks.
