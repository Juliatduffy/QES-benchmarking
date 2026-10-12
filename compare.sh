#!/bin/bash
#SBATCH -M granite                      
#SBATCH --nodes=1                       
#SBATCH --ntasks=1                      
#SBATCH --partition=coe-class-grn       
#SBATCH --qos=coe-class-shortbatch-grn  
#SBATCH --account=cs6230                
#SBATCH --cpus-per-task=64              
#SBATCH --exclusive                     
#SBATCH -t 0:10:00                      
#SBATCH -o ./Testing/logs/compare.out

OPTIMIZATION=$1     # $1 is the first word after script name, ex: bugfix
NTRIALS=10          

CASE=~/QES/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml   
BASELINE=~/QES/build-public/qesWinds/qesWinds                     
OPTIMIZED=~/QES/build-cpu/qesWinds/qesWinds                       

cd ~/QES
source QES-Public/CHPC/loadmodules_QES.sh
export OMP_PLACES=cores OMP_PROC_BIND=close    # keeps each thread on its own core    

LOG=~/QES/Testing/logs/$(date +%Y-%m-%d_%H-%M-%S)_${OPTIMIZATION}.log
echo "Summary will be written to: $LOG"    
exec > "$LOG" 2>&1


# collect_results "<program and flags>": runs QES once and collects its results:
# TIME (the solver's own elapsed time, in seconds) and RESID (the residual it printed)
collect_results() {
  OUTPUT=$($1 -q $CASE 2>/dev/null)
  RESID=$(echo "$OUTPUT" | awk '/Residual after/ {print $NF}')
  TIME=$(echo "$OUTPUT" | awk '/Residual after/ {getline; print $3}')
}

echo "QES-Public vs $OPTIMIZATION"
echo "Input wind field: $CASE"
echo "Node $SLURMD_NODENAME, job $SLURM_JOB_ID, $(date)"

for NT in 1 16 64; do
  export OMP_NUM_THREADS=$NT   
  # warmup runs, not timed; the optimization's also saves its u, v, w
  $BASELINE -q $CASE > /dev/null 2>&1
  OPT_OUT=~/QES/Testing/runs/cpu/$OPTIMIZATION/okc_${NT}threads
  mkdir -p $OPT_OUT
  if ! $OPTIMIZED -c $OPTIMIZATION -q $CASE -o $OPT_OUT/okc -w > /dev/null 2>&1; then
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
for NT in 1 16 64; do
  BASE_FILE=~/QES/Testing/runs/public/okc_${NT}threads/okc_windsWk.nc
  OPT_FILE=~/QES/Testing/runs/cpu/$OPTIMIZATION/okc_${NT}threads/okc_windsWk.nc
  if nccmp -d -S -f -q -v u,v,w $BASE_FILE $OPT_FILE; then
    echo "  $NT thread(s): IDENTICAL"
  else
    echo "  $NT thread(s): DIFFERENT (see the table above)"
  fi
done
