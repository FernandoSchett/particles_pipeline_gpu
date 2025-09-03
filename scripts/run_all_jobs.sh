cd "$(dirname "$0")/.."

sh ./compile.sh

# call other experiments
sh ./run_cpu_jobs.sh
sh ./run_gpu_jobs.sh