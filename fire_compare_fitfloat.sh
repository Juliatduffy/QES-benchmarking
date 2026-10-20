#!/bin/bash
# Compares a fire run with the original QES GPU solver (QES-Public qesFire -s 3) against FitFloat
# variants (QES-FitFloat qesFire -s 3 -c <variant>): fire-loop time, wind-solve and plume time,
# whole run, and whether every variable in the fire output is the same.
#
# Run on the PC, from ~/QES (after profile_gpu.sh pc has built QES-Public):
#   bash fire_compare_fitfloat.sh fire                 fire.xml: 400x400x20 cells of 1 m, variant ff-en4
#   bash fire_compare_fitfloat.sh tall                 Testing/inputs/fire_tall.xml: the same fire in an 80 m tall box
#   bash fire_compare_fitfloat.sh tall ff-en4 ff-ctl   other variants (names as in wind_compare_fitfloat.sh)
#
# Both inputs have 1 m cells and flat ground, so ff-en4 stores e..n exactly (lossless).
# qesFire writes its output every fire step (fire ~2.5 GB per run, tall ~10 GB); at most two runs'
# files are on disk at once. One tall run takes a few minutes; the whole script ~20 min.
#
# Controls: the original runs twice (same answer every run?).
# Fire loop = QES's own fire-step timers added up. It leaves out setup, the first wind solve
# (where FitFloat packs e..n) and writing the output each step; "whole run" includes everything.

CASE_NAME=$1
shift
VARIANTS="$*"
if [ -z "$VARIANTS" ]; then
  VARIANTS="ff-en4"
fi
if [ "$CASE_NAME" = fire ]; then
  CASE=$HOME/QES/QES-Public/data/InputFiles/fire.xml; NEED_GB=6
elif [ "$CASE_NAME" = tall ]; then
  CASE=$HOME/QES/Testing/inputs/fire_tall.xml; NEED_GB=22
else
  echo "usage: bash fire_compare_fitfloat.sh fire|tall [variants]"
  exit 1
fi
NTRIALS=3

BASELINE=$HOME/QES/build-gpu-public-sm89/qesFire/qesFire   # built by profile_gpu.sh pc
FF_BUILD=$HOME/QES/build-gpu-fitfloat-sm89
FITFLOAT=$FF_BUILD/qesFire/qesFire
WORK=$HOME/gpu_profile/fitfloat_fire
export OMP_NUM_THREADS=8

# the output check needs python's numpy and netCDF4
if ! python3 -c "import numpy, netCDF4" 2>/dev/null; then
  echo "first run: sudo apt install -y python3-numpy python3-netcdf4"
  exit 1
fi
if [ ! -x $BASELINE ]; then
  echo "QES-Public GPU build missing: run 'bash profile_gpu.sh pc' first"
  exit 1
fi

# qesFire opens ../data/FireFiles/HeatPot.nc from the folder it starts in, so start it in $WORK/run
mkdir -p $WORK/run $WORK/out $HOME/QES/Testing/logs
ln -sfn $HOME/QES/QES-Public/data $WORK/data
ln -sfn $HOME/QES/QES-Public/data/FireFiles $WORK/FireFiles
FREE_GB=$(df --output=avail -BG $WORK | tail -1 | tr -dc 0-9)
if (( FREE_GB < NEED_GB )); then
  echo "$CASE_NAME needs about $NEED_GB GB free in $WORK, only $FREE_GB GB left"
  exit 1
fi

# everything printed after this point goes into the log
LOG=$HOME/QES/Testing/logs/$(date +%Y-%m-%d_%H-%M-%S)_fitfloat_fire_${CASE_NAME}_pc.log
echo "Log: $LOG"
exec > "$LOG" 2>&1

echo "Fire run ($CASE_NAME: $(basename $CASE)), original GPU solver vs FitFloat variants: $VARIANTS (control: original 2nd run)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)   date: $(date '+%Y-%m-%d %H:%M')   commit: $(git -C $HOME/QES rev-parse --short HEAD)"

# build QES-FitFloat's qesFire with CUDA (make only recompiles what changed; output goes to build.txt)
if [ ! -d $FF_BUILD ]; then
  cmake -S $HOME/QES/QES-FitFloat -B $FF_BUILD -DENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=89 > $WORK/build.txt 2>&1
fi
make -C $FF_BUILD -j8 qesFire >> $WORK/build.txt 2>&1 || { echo "build failed, see $WORK/build.txt"; exit 1; }

# command_for <version> <output basename>: prints the command that runs that version
command_for() {
  if [ "$1" = original ]; then
    echo "$BASELINE -q $CASE -s 3 -o $2 -b"
  else
    echo "$FITFLOAT -q $CASE -s 3 -c $1 -o $2 -b"
  fi
}

# run_once <version> <output basename>: runs qesFire once (QES's messages go to <basename>.txt); sets
# WHOLE (whole run), LOOP (fire steps), WIND (all wind solves), POT (plume winds), PACK (FitFloat packing)
run_once() {
  local START=$(date +%s.%N)
  cd $WORK/run
  $(command_for $1 $2) > $2.txt 2>&1
  local STATUS=$?
  WHOLE=$(awk -v s=$START -v e=$(date +%s.%N) 'BEGIN {print e - s}')
  read LOOP WIND POT <<< $(awk '/^\[/ {stage = $0}
    /elapsed time/ {
      if (stage ~ /Current time/) loop += $(NF-1)
      else if (stage ~ /Residual after/) wind += $(NF-1)
      else if (stage ~ /GPU POTENTIAL/) pot += $(NF-1)
    }
    END {print loop + 0, wind + 0, pot + 0}' $2.txt)
  PACK=$(awk '/packing time/ {print $(NF-1)}' $2.txt)
  PACK=${PACK:-0}
  return $STATUS
}

# compare_fire <fireOutput.nc> <fireOutput.nc>: "identical", or which variables differ and by how much.
# Reads one time step (first index) at a time, so a 10 GB file never has to fit in memory.
compare_fire() {
python3 - "$1" "$2" <<'PYEOF'
import sys
import numpy as np
from netCDF4 import Dataset
a, b = Dataset(sys.argv[1]), Dataset(sys.argv[2])
a.set_auto_mask(False)
b.set_auto_mask(False)
lines = []
for name, va in a.variables.items():
    if name not in b.variables:
        lines.append(f"    {name}: missing in the second file")
        continue
    vb = b.variables[name]
    if va.shape != vb.shape:
        lines.append(f"    {name}: shape {va.shape} vs {vb.shape}")
        continue
    ndiff, nbig, maxd = 0, 0, 0.0
    for c in (range(va.shape[0]) if va.ndim >= 2 else [Ellipsis]):
        x, y = np.asarray(va[c]), np.asarray(vb[c])
        if x.dtype.kind not in "fiu":
            ndiff += 0 if np.array_equal(x, y) else 1
            continue
        if np.array_equal(x, y, equal_nan=(x.dtype.kind == "f")):
            continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        ndiff += np.count_nonzero(d != 0)
        nbig += np.count_nonzero(d > 0.01)
        maxd = max(maxd, float(np.nanmax(d)))
    if ndiff:
        lines.append(f"    {name}: {ndiff} values differ, max |diff| {maxd:.2e}, {nbig} by more than 0.01")
print("\n".join(lines) if lines else "    identical (every variable, every step)")
PYEOF
}

declare -A SUM_LOOP MIN_LOOP MAX_LOOP SUM_WIND SUM_POT SUM_WHOLE SUM_PACK ERROR

# warm-up and correctness: the original twice (does it give the same answer every run?), then each
# variant once; each file is compared right away and deleted, so at most two runs' files are on disk
ORIG=$WORK/out/${CASE_NAME}_original
if ! run_once original $ORIG || ! run_once original ${ORIG}_rerun; then
  echo "the original failed, see $ORIG.txt and ${ORIG}_rerun.txt"
  exit 1
fi
REPEAT=$(compare_fire ${ORIG}_fireOutput.nc ${ORIG}_rerun_fireOutput.nc)
rm -f ${ORIG}_rerun_fireOutput.nc
RAN=""
for V in $VARIANTS; do
  OUT=$WORK/out/${CASE_NAME}_$V
  if ! run_once $V $OUT; then
    echo "$V failed, see $OUT.txt"
    continue
  fi
  RAN="$RAN $V"
  ERROR[$V]=$(compare_fire ${ORIG}_fireOutput.nc ${OUT}_fireOutput.nc)
  rm -f ${OUT}_fireOutput.nc
done
rm -f ${ORIG}_fireOutput.nc
STEPS=$(grep -c "Fire step completed" $ORIG.txt)

# timed runs: the original and each variant in turn; each run's output is deleted right after it
for NAME in original $RAN; do
  SUM_LOOP[$NAME]=0; MIN_LOOP[$NAME]=1000000; MAX_LOOP[$NAME]=0
  SUM_WIND[$NAME]=0; SUM_POT[$NAME]=0; SUM_WHOLE[$NAME]=0; SUM_PACK[$NAME]=0
done
for ((trial = 1; trial <= NTRIALS; trial++)); do
  for NAME in original $RAN; do
    run_once $NAME $WORK/out/timing || echo "trial $trial: $NAME failed, see $WORK/out/timing.txt"
    rm -f $WORK/out/timing_fireOutput.nc
    SUM_LOOP[$NAME]=$(echo "${SUM_LOOP[$NAME]} + $LOOP" | bc -l)
    SUM_WIND[$NAME]=$(echo "${SUM_WIND[$NAME]} + $WIND" | bc -l)
    SUM_POT[$NAME]=$(echo "${SUM_POT[$NAME]} + $POT" | bc -l)
    SUM_WHOLE[$NAME]=$(echo "${SUM_WHOLE[$NAME]} + $WHOLE" | bc -l)
    SUM_PACK[$NAME]=$(echo "${SUM_PACK[$NAME]} + $PACK" | bc -l)
    if (( $(echo "$LOOP < ${MIN_LOOP[$NAME]}" | bc -l) )); then MIN_LOOP[$NAME]=$LOOP; fi
    if (( $(echo "$LOOP > ${MAX_LOOP[$NAME]}" | bc -l) )); then MAX_LOOP[$NAME]=$LOOP; fi
  done
done

echo
echo "time (s), $NTRIALS alternating trials, $STEPS fire steps per run:"
echo "  fire loop = all fire steps (wind solve + plume + level set + move), without setup, packing or writing output"
printf "  %-9s %9s %9s %9s %9s %9s %9s %8s %10s %10s\n" version "loop min" "loop max" "loop avg" wind plume whole packing "loop spdup" "whole spdup"
avg() { echo "$1 / $NTRIALS" | bc -l; }
for NAME in original $RAN; do
  printf "  %-9s %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f %8.2f %9.2fx %9.2fx\n" $NAME ${MIN_LOOP[$NAME]} ${MAX_LOOP[$NAME]} \
    $(avg ${SUM_LOOP[$NAME]}) $(avg ${SUM_WIND[$NAME]}) $(avg ${SUM_POT[$NAME]}) $(avg ${SUM_WHOLE[$NAME]}) $(avg ${SUM_PACK[$NAME]}) \
    $(echo "${SUM_LOOP[original]} / ${SUM_LOOP[$NAME]}" | bc -l) $(echo "${SUM_WHOLE[original]} / ${SUM_WHOLE[$NAME]}" | bc -l)
done

echo "what each variant stored (from QES's own output):"
for V in $RAN; do
  echo "  $V:"
  grep "\[FitFloat\]" $WORK/out/${CASE_NAME}_$V.txt | grep -v "packing time" | sed 's/\[FitFloat\]\s*/    /'
done

echo "same answer every run? (the original's 2nd run vs its 1st; must be identical):"
echo "$REPEAT"

echo "fire output vs the original (every variable, every step; must be identical for a lossless variant):"
for V in $RAN; do
  echo "  $V:"
  echo "${ERROR[$V]}"
done
