#!/bin/bash
#SBATCH --job-name=nandos_cool_merg_algthm_exp       
#SBATCH --output=nandos_cool_merg_algthm_exp.out     
#SBATCH --error=nandos_cool_merg_algthm_exp.err     
#SBATCH --nodes=1                 
#SBATCH --ntasks=256             
#SBATCH --cpus-per-task=1         
#SBATCH --time=02:00:00
#SBATCH --partition=masid        
#SBATCH --nodelist=jrc[0737-0832]

source ./load_modules.sh

AVAILABLE=$(nproc)
echo "Available processes: $AVAILABLE."

cd "$(dirname "$0")/.."

rm -rf build
rm -f results.csv
mkdir build
cd build

cmake ..
cmake --build .

for np in 4 8 16 32 64 128 156
do
    for i in 1 2 3 4 5
    do
        echo "Executing with $np processes..."
        srun -n $np ./p_sfc_exe 8
    done
done
