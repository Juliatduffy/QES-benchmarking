#!/bin/bash
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn     
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH -t 1:30:00
#SBATCH -o ./Testing/logs/fire_compare.out


FLAGS=""
if [ "$1" = -f ]; then
  FLAGS="-f parallel"
  shift                                     
fi
OPTIMIZATION=$1
FLAGS="$FLAGS -c $OPTIMIZATION"
CASE_NAME=${2:-short}
NTRIALS=3
THREADS="16 64"
if [ "$CASE_NAME" = tiny ]; then NTRIALS=1; fi

LABEL=$(echo "$FLAGS" | tr -d '-' | tr ' ' '_')  
BASELINE=~/QES/build-public/qesFire/qesFire
OPTIMIZED=~/QES/build-cpu/qesFire/qesFire
FIRE=/scratch/general/vast/$USER/qes-fire    
CASE=../inputs/fire_${CASE_NAME}.xml
OUT=$FIRE/out
mkdir -p $OUT

cd ~/QES
source QES-Public/CHPC/loadmodules_QES.sh
export OMP_PLACES=cores OMP_PROC_BIND=close

LOG=~/QES/Testing/logs/$(date +%Y-%m-%d_%H-%M-%S)_fire_${LABEL}_${CASE_NAME}.log
echo "Summary will be written to: $LOG"
exec > "$LOG" 2>&1

cd $FIRE/run

collect_results() {
  RESULTS=$($1 -q $CASE -o $2 -b 2>/dev/null | awk '
    /POTENTIAL/           { in_pot = 1 }
    /Residual after/      { in_sol = 1; next }
    /Fire step completed/ { in_fire = 1 }
    /elapsed time/ {
      if (in_pot)       { pot  += $3; in_pot = 0 }
      else if (in_sol)  { sol  += $3; in_sol = 0 }
      else if (in_fire) { fire += $3; in_fire = 0 }
    }
    END { printf "%f %f %f\n", fire, pot, sol }')
  read FIRE_T POT_T SOL_T <<< "$RESULTS"
}

echo "QES-Public qesFire vs QES-CPU qesFire $FLAGS"
echo "Input: $FIRE/inputs/fire_${CASE_NAME}.xml"
echo "Node $SLURMD_NODENAME, job $SLURM_JOB_ID, $(date)"

for NT in $THREADS; do
  export OMP_NUM_THREADS=$NT
  BASE_SUM=0; BASE_POT=0; BASE_SOL=0
  OPT_SUM=0;  OPT_POT=0;  OPT_SOL=0

  for ((trial = 1; trial <= NTRIALS; trial++)); do
    collect_results "$BASELINE" $OUT/public_${NT}t
    BASE_SUM=$(echo "$BASE_SUM + $FIRE_T" | bc -l)
    BASE_POT=$(echo "$BASE_POT + $POT_T" | bc -l)
    BASE_SOL=$(echo "$BASE_SOL + $SOL_T" | bc -l)

    collect_results "$OPTIMIZED $FLAGS" $OUT/${LABEL}_${NT}t
    if [ "$FIRE_T" = "0.000000" ]; then
      echo "ERROR: QES-CPU qesFire $FLAGS failed to run (try it by hand to see why)"
      exit 1
    fi
    OPT_SUM=$(echo "$OPT_SUM + $FIRE_T" | bc -l)
    OPT_POT=$(echo "$OPT_POT + $POT_T" | bc -l)
    OPT_SOL=$(echo "$OPT_SOL + $SOL_T" | bc -l)
  done

  echo
  echo "$NT threads, average of $NTRIALS alternating trials (seconds):"
  printf "  %-24s fire loop %8.2f   potential %8.2f   wind solver %7.2f\n" "QES-Public" \
    $(echo "$BASE_SUM / $NTRIALS" | bc -l) $(echo "$BASE_POT / $NTRIALS" | bc -l) $(echo "$BASE_SOL / $NTRIALS" | bc -l)
  printf "  %-24s fire loop %8.2f   potential %8.2f   wind solver %7.2f\n" "$FLAGS" \
    $(echo "$OPT_SUM / $NTRIALS" | bc -l) $(echo "$OPT_POT / $NTRIALS" | bc -l) $(echo "$OPT_SOL / $NTRIALS" | bc -l)
  printf "  Speedup (fire loop): %.2fx\n" $(echo "$BASE_SUM / $OPT_SUM" | bc -l)
done

module load nccmp/1.9.1.0 > /dev/null 2>&1
echo
echo "Correctness: every variable of the fire output, compared with QES-Public's:"
for NT in $THREADS; do
  if nccmp -d -S -f -q $OUT/public_${NT}t_fireOutput.nc $OUT/${LABEL}_${NT}t_fireOutput.nc; then
    echo "  $NT threads: IDENTICAL"
  else
    echo "  $NT threads: DIFFERENT (see the table above)"
  fi
  rm -f $OUT/public_${NT}t_* $OUT/${LABEL}_${NT}t_*  
done
