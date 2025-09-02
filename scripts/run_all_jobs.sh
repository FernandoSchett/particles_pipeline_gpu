cd "$(dirname "$0")/.."

# build project
rm -rf build
rm -f results.csv
mkdir build
cd build

cmake ..
cmake --build .

# call other experiments
sh ../scripts/run_cpu_jobs.sh
sh ../scripts/run_gpu_jobs.sh