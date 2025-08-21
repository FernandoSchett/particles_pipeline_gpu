#!/bin/bash
#SBATCH --job-name=nandos_cool_merg_algthm_exp       
#SBATCH --output=nandos_cool_merg_algthm_exp.out     
#SBATCH --error=nandos_cool_merg_algthm_exp.err     
#SBATCH --nodes=1                 
#SBATCH --ntasks=128             
#SBATCH --cpus-per-task=1         
#SBATCH --time=01:00:00
#SBATCH --partition=masid        

source ./load_modules.sh

cd "$(dirname "$0")/.."

rm -rf build
rm -f results.csv
mkdir build
cd build

cmake ..
cmake --build .

for np in 4 8 16
do
    for i in 1 2 3 4 5
    do
        echo "Executing with $np processes..."
        srun -n $np ./p_sfc_exe 8
    done
done
