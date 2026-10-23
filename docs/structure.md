# Project structure

```text
particles_pipeline_gpu/
├── .env                         # CUDA compiler path (CUDACXX)
├── CMakeLists.txt               # Dependencies, build options, and CTest
├── README.md                    # Build and execution instructions
├── requirements.txt             # Python dependencies
├── src/
│   ├── CMakeLists.txt
│   ├── cpu_main_particles.cpp   # CPU + MPI pipeline entry point
│   └── gpu_mpi_main_particles.cu # CUDA + MPI pipeline entry point
├── libs/
│   ├── CMakeLists.txt           # Library subdirectories and target aliases
│   ├── common/
│   │   ├── CMakeLists.txt
│   │   ├── include/            # Shared types, I/O, domain, logging, and utilities
│   │   ├── src/                # Particle I/O, PEPC reader, domain fitting, CSV logging
│   │   └── tests/
│   │       └── particle_domain_tests.cpp
│   ├── cpu/
│   │   ├── CMakeLists.txt
│   │   ├── include/            # CPU particles and hashed octree interfaces
│   │   ├── src/                # Generation, Morton partitioning, distributed hashed octree
│   │   └── tests/
│   │       ├── CMakeLists.txt
│   │       └── distributed_hashed_octree_tests.cpp
│   └── gpu/
│       ├── CMakeLists.txt
│       ├── include/            # CUDA particle and octree interfaces (.hcu)
│       └── src/                # Particle kernels, PEPC upload, MPI exchange, cornerstone octree
├── scripts/
│   ├── compile.sh              # Load .env and rebuild project
│   ├── setup_py_env.sh         # Create Python environment and install dependencies
│   ├── load_modules_cpu.sh     # Cluster modules for CPU execution
│   ├── load_modules_gpu.sh     # Cluster modules for GPU execution
│   ├── run_all_jobs.sh         # Build and submit CPU/GPU experiments
│   ├── run_cpu_jobs.sh         # Submit CPU scaling jobs through Slurm
│   ├── run_gpu_jobs.sh         # Submit GPU scaling jobs through Slurm
│   ├── run_cpu_experiment.sh   # Repeat CPU executions
│   ├── run_gpu_experiment.sh   # Repeat GPU executions
│   └── tests/                  # Slurm test launchers and output validation
├── py_apps/
│   ├── gen_results.py          # Scaling plots from experiment CSVs
│   ├── visualise.py            # Interactive particle plots
│   ├── tree_visualise.py       # CPU hashed octree visualization
│   ├── verify_par_file.py      # Particle ordering and load balance checks
│   ├── verify_tree_file.py     # CPU tree topology and coverage checks
│   ├── tree_reference.py       # Independent CPU/GPU trees and particle cross-checks
│   ├── run_tree_correctness.py # MPI end-to-end tests with PEPC edge-case fixtures
│   ├── tests/                 # Validator tests, including deliberate corruption
│   └── verify_global_tree_file.py # GPU cornerstone and topology checks
├── docs/
│   ├── structure.md            # This directory map
│   ├── tree_correctness.md     # Correctness coverage and execution commands
│   └── tree_construction.pdf   # Tree construction reference
├── results/
│   ├── debug/                 # Direct-run particle/tree files and timing CSVs (generated)
│   ├── tests/                 # Slurm test outputs (generated)
│   ├── strong/                # Strong-scaling figures
│   ├── weak/                  # Weak-scaling figures
│   └── particles/mpi/          # PEPC input snapshot and parameter file
├── third_party/
│   ├── CMakeLists.txt
│   └── random123/             # Random123 submodule for particle generation
└── build/                     # Generated CMake files, libraries, executables, and tests
```

## Build outputs and data files

- `build/src/p_sfc_exe`: CPU executable.
- `build/src/gpu_mpi_p_sfc_exe`: GPU executable, enabled by `P_SFC_BUILD_GPU`.
- `libs/common/tests/` and `libs/cpu/tests/`: C++ tests registered with CTest when `BUILD_TESTING` is enabled.
- `scripts/tests/`: cluster execution scripts; these also run Python validators on generated files.
- `.par`, `.tree`, and `.gtree`: particle, CPU tree, and GPU tree outputs written under `results/debug/` by default, or `P_SFC_OUTPUT_DIR`. Slurm tests use `results/tests/`.
- `visualise.py` searches the project root, `build/`, the current working directory, `results/debug/`, `results/tests/`, and `P_SFC_OUTPUT_DIR` for `particle_file*.par`.
- `results_strong.csv` and `results_weak.csv`: timing logs written in the output directory. `gen_results.py` defaults to `results/debug/` and accepts `--csv-dir`.
- `results/particles/mpi/particle_000000000001.mpi`: bundled PEPC snapshot with 10,000 particles. Its `.mpi.h` companion contains PEPC parameters in text form, not a C++ header.

The shared implementation lives in `libs/common/`; CPU and GPU algorithms live in their respective libraries. The obsolete `libs/helper/` directory has been removed.
