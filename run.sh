#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn          
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=64
#SBATCH --exclusive 
#SBATCH -t 0:30:00
#SBATCH -o ./Testing/logs/run.out
echo "*** Assigned Granite Node: " $SLURMD_NODENAME 
cd ~/QES
source QES-Public/CHPC/loadmodules_QES.sh
export OMP_PLACES=cores OMP_PROC_BIND=close
for CASE_NAME in okc slc; do
CASE=~/QES/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml
THREADS="1 16 64"
if [ "$CASE_NAME" = slc ]; then
  CASE=~/QES/QES-Public/tests/runTestsFiles/winds_SaltLakeCity.xml
  THREADS="16 64"
fi

for NT in $THREADS; do
  export OMP_NUM_THREADS=$NT
  mkdir -p ~/QES/Testing/runs/public/${CASE_NAME}_${NT}threads
  ./build-public/qesWinds/qesWinds \
      -q $CASE \
      -o ~/QES/Testing/runs/public/${CASE_NAME}_${NT}threads/${CASE_NAME} \
      -w
done
done  


