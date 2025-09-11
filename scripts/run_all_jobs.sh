cd "$(dirname "$0")"

source ./load_modules_cpu.sh

sh ./compile.sh

# call other experiments
sh ./run_cpu_jobs.sh
sh ./run_gpu_jobs.sh
