| Item                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `build/`              | Compiled output directory (CMake build artifacts, binaries, object files). |
| `docs/`               | Documentation files for the project (design notes, reports, references).   |
| `gen_results.py`      | Python script to parse/aggregate raw experiment results into CSVs.          |
| `gen_results.ipynb`   | Jupyter notebook for generating and analyzing results interactively.        |
| `gen_results_save.ipynb` | Jupyter notebook variant for saving experiment outputs and figures.      |
| `libs/`               | Project libraries (helper modules, reusable code for CPU/GPU, utilities).  |
| `requirements.txt`    | Python dependencies for analysis/visualization scripts.                     |
| `scripts/`            | Bash/Slurm scripts to launch experiments on clusters.                       |
| `tests/`              | Unit tests and validation cases for core components.                        |
| `visualise.py`        | Python script to plot and visualize performance metrics.                    |
| `CMakeLists.txt`      | Main build configuration (CMake project definition and linking).            |
| `README.md`           | Project overview and usage instructions.                                    |
| `results/`            | Collected experiment outputs (CSV logs, runtime data).                      |
| `src/`                | Main C++ source code (simulation kernels, MPI+CUDA execution, algorithms).  |
| `third_party/`        | External dependencies included in the repository.                           |
