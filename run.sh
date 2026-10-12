#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shortbatch-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=64
#SBATCH --exclusive 
#SBATCH -t 0:10:00
#SBATCH -o ./Testing/logs/run.out
echo "*** Assigned Granite Node: " $SLURMD_NODENAME 
cd ~/QES
source QES-Public/CHPC/loadmodules_QES.sh
export OMP_PLACES=cores OMP_PROC_BIND=close
for NT in 1 16 64; do
  export OMP_NUM_THREADS=$NT
  mkdir -p ~/QES/Testing/runs/public/okc_${NT}threads
  ./build-public/qesWinds/qesWinds \
      -q ~/QES/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml \
      -o ~/QES/Testing/runs/public/okc_${NT}threads/okc \
      -w
done


