#!/bin/bash
#SBATCH --job-name=nandos_cool_merg_algthm_exp
#SBATCH --output=nandos_cool_merg_algthm_exp_%j.out
#SBATCH --error=nandos_cool_merg_algthm_exp_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=dc-cpu
#SBATCH --account=gsp25
     
source ./load_modules.sh

cd "$(dirname "$0")/.."

rm -rf build
rm -f results.csv
mkdir build
cd build

cmake ..
cmake --build .

NP=$1

echo "Running with $NP processes"

for i in 1 2 3 4 5
do
    echo "Execution #$i with $NP processes..."
    srun -n $NP ./p_sfc_exe 2
done
