#!/bin/bash
#SBATCH -M granite                      
#SBATCH --nodes=1                       
#SBATCH --ntasks=1                      
#SBATCH --partition=coe-class-grn       
#SBATCH --qos=coe-students-grn         
#SBATCH --account=cs6230                
#SBATCH --cpus-per-task=64              
#SBATCH --exclusive                     
#SBATCH -t 0:30:00
#SBATCH -o ./Testing/logs/wind_compare.out

OPTIMIZATION=$1     # $1 is the first word after script name, ex: bugfix


BASELINE=~/QES/build-public/qesWinds/qesWinds                     
OPTIMIZED=~/QES/build-cpu/qesWinds/qesWinds                       

cd ~/QES
source QES-Public/CHPC/loadmodules_QES.sh
export OMP_PLACES=cores OMP_PROC_BIND=close    # keeps each thread on its own core    

LOG=~/QES/Testing/logs/$(date +%Y-%m-%d_%H-%M-%S)_${OPTIMIZATION}.log
echo "Summary will be written to: $LOG"    
exec > "$LOG" 2>&1


collect_results() {
  OUTPUT=$($1 -q $CASE 2>/dev/null)
  RESID=$(echo "$OUTPUT" | awk '/Residual after/ {print $NF}')
  TIME=$(echo "$OUTPUT" | awk '/Residual after/ {getline; print $3}')
}

echo "QES-Public vs $OPTIMIZATION"
echo "Node $SLURMD_NODENAME, job $SLURM_JOB_ID, $(date)"

for CASE_NAME in okc slc; do
if [ "$CASE_NAME" = slc ]; then
  CASE=~/QES/QES-Public/tests/runTestsFiles/winds_SaltLakeCity.xml; THREADS="16 64"; NTRIALS=3
else
  CASE=~/QES/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml; THREADS="1 16 64"; NTRIALS=10
fi
echo
echo "===== $CASE_NAME: $CASE ====="

for NT in $THREADS; do
  export OMP_NUM_THREADS=$NT   
  $BASELINE -q $CASE > /dev/null 2>&1
  OPT_OUT=~/QES/Testing/runs/cpu/$OPTIMIZATION/${CASE_NAME}_${NT}threads
  mkdir -p $OPT_OUT
  if ! $OPTIMIZED -c $OPTIMIZATION -q $CASE -o $OPT_OUT/$CASE_NAME -w > /dev/null 2>&1; then
    echo "ERROR: '$OPTIMIZATION' failed to run."
    exit 1
  fi

  BASE_MIN=1000000; BASE_MAX=0; BASE_SUM=0
  OPT_MIN=1000000;  OPT_MAX=0;  OPT_SUM=0

  for ((trial = 1; trial <= NTRIALS; trial++)); do
    collect_results "$BASELINE"
    BASE_RESID=$RESID
    BASE_SUM=$(echo "$BASE_SUM + $TIME" | bc -l)
    if (( $(echo "$TIME < $BASE_MIN" | bc -l) )); then BASE_MIN=$TIME; fi
    if (( $(echo "$TIME > $BASE_MAX" | bc -l) )); then BASE_MAX=$TIME; fi

    collect_results "$OPTIMIZED -c $OPTIMIZATION"
    OPT_RESID=$RESID
    OPT_SUM=$(echo "$OPT_SUM + $TIME" | bc -l)
    if (( $(echo "$TIME < $OPT_MIN" | bc -l) )); then OPT_MIN=$TIME; fi
    if (( $(echo "$TIME > $OPT_MAX" | bc -l) )); then OPT_MAX=$TIME; fi
  done

  BASE_AVG=$(echo "$BASE_SUM / $NTRIALS" | bc -l)
  OPT_AVG=$(echo "$OPT_SUM / $NTRIALS" | bc -l)
  SPEEDUP=$(echo "$BASE_AVG / $OPT_AVG" | bc -l)

  if [ "$BASE_RESID" = "$OPT_RESID" ]; then
    RESID_CHECK="match"
  else
    RESID_CHECK="different"
  fi

  echo
  echo "Solver time in seconds, $NTRIALS alternating trials of each:"
  echo "$NT thread(s):"
  printf "  QES-Public:  Min: %.4f  Max: %.4f  Avg: %.4f\n" $BASE_MIN $BASE_MAX $BASE_AVG
  printf "  %s:  Min: %.4f  Max: %.4f  Avg: %.4f\n" $OPTIMIZATION $OPT_MIN $OPT_MAX $OPT_AVG
  printf "  Speedup (avg): %.2fx\n" $SPEEDUP
  echo "  Residual: QES-Public $BASE_RESID, $OPTIMIZATION $OPT_RESID ($RESID_CHECK)"
done

# Correctness: at each thread count, compare optimization's u, v, w with
# QES-Public's from the same thread count.
module load nccmp/1.9.1.0 > /dev/null 2>&1

echo
echo "Correctness: $OPTIMIZATION's u, v, w compared with QES-Public's, every value:"
for NT in $THREADS; do
  BASE_FILE=~/QES/Testing/runs/public/${CASE_NAME}_${NT}threads/${CASE_NAME}_windsWk.nc
  OPT_FILE=~/QES/Testing/runs/cpu/$OPTIMIZATION/${CASE_NAME}_${NT}threads/${CASE_NAME}_windsWk.nc
  if nccmp -d -S -f -q -v u,v,w $BASE_FILE $OPT_FILE; then
    echo "  $NT thread(s): IDENTICAL"
  else
    echo "  $NT thread(s): DIFFERENT (see the table above)"
  fi
  if [ "$CASE_NAME" = slc ]; then rm -f $OPT_FILE ${OPT_FILE%Wk.nc}Out.nc; fi
done
# reload QES's exact modules before the next case
source ~/QES/QES-Public/CHPC/loadmodules_QES.sh > /dev/null 2>&1
done  

