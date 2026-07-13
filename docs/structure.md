| Item                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `build/`              | Compiled output directory (CMake build artifacts, binaries, object files). |
| `docs/`               | Documentation files for the project (design notes, reports, references).   |
| `py_apps/`            | Python applications for result analysis and particle visualization.         |
| `py_apps/gen_results.py` | Reads experiment CSVs and generates scaling plots under `results/`.       |
| `py_apps/visualise.py` | Reads particle binaries from `build/` and displays interactive 3D plots.    |
| `py_apps/verify_par_file.py` | Validates ordering and load balance in particle `.par` files.          |
| `gen_results.ipynb`   | Jupyter notebook for generating and analyzing results interactively.        |
| `gen_results_save.ipynb` | Jupyter notebook variant for saving experiment outputs and figures.      |
| `libs/`               | Project libraries (helper modules, reusable code for CPU/GPU, utilities).  |
| `requirements.txt`    | Python dependencies for analysis/visualization scripts.                     |
| `scripts/`            | Bash/Slurm scripts to launch experiments on clusters.                       |
| `tests/`              | Unit tests and validation cases for core components.                        |
| `CMakeLists.txt`      | Main build configuration (CMake project definition and linking).            |
| `README.md`           | Project overview and usage instructions.                                    |
| `results/`            | Collected experiment outputs (CSV logs, runtime data).                      |
| `src/`                | Main C++ source code (simulation kernels, MPI+CUDA execution, algorithms).  |
| `third_party/`        | External dependencies included in the repository.                           |
