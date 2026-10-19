#!/bin/bash
# Compares the original QES GPU solver (QES-Public, -s 3) with FitFloat variants
# (QES-FitFloat, -s 3 -c <variant>) on OKC and SLC: solver time and the error in u, v, w.
#
# Run on the PC, from ~/QES (after profile_gpu.sh pc has built QES-Public):
#   bash wind_compare_fitfloat.sh                    all variants
#   bash wind_compare_fitfloat.sh ff-ctl ff-en4      only these
#
# Variants (what is stored as FitFloat; the math stays float):
#   ff-ctl      e..n in 32 bits (1+8+23): same values, no bytes saved -> the cost of unpacking alone
#   ff-en16     e..n in 16 bits (1+8+7)
#   ff-en4      e..n in  4 bits (1+3+0); falls back to float e..n if a value is out of range (OKC)
#   ff-r16      R in 16 bits (1+8+7)
#   ff-en16r16  e..n and R in 16 bits
#
# Controls, always run:
#   ff-build    the QES-FitFloat program WITHOUT -c = the original solver code in the FitFloat build.
#               Must be identical to the original and just as fast; if not, the build differs, not FitFloat.
#   2nd run     every version runs twice; the two answers must be identical (same input, same code).

VARIANTS="$*"
if [ -z "$VARIANTS" ]; then
  VARIANTS="ff-ctl ff-en16 ff-en4 ff-r16 ff-en16r16"
fi

BASELINE=$HOME/QES/build-gpu-public-sm89/qesWinds/qesWinds   # built by profile_gpu.sh pc
FF_BUILD=$HOME/QES/build-gpu-fitfloat-sm89
FITFLOAT=$FF_BUILD/qesWinds/qesWinds
WORK=$HOME/gpu_profile/fitfloat
export OMP_NUM_THREADS=8

# the error check needs python's numpy and netCDF4
if ! python3 -c "import numpy, netCDF4" 2>/dev/null; then
  echo "first run: sudo apt install -y python3-numpy python3-netcdf4"
  exit 1
fi
if [ ! -x $BASELINE ]; then
  echo "QES-Public GPU build missing: run 'bash profile_gpu.sh pc' first"
  exit 1
fi

# everything printed after this point goes into the log
mkdir -p $WORK $HOME/QES/Testing/logs
LOG=$HOME/QES/Testing/logs/$(date +%Y-%m-%d_%H-%M-%S)_fitfloat_compare_pc.log
echo "Log: $LOG"
exec > "$LOG" 2>&1

echo "Original GPU solver vs FitFloat variants: $VARIANTS (controls: ff-build, 2nd run)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)   date: $(date '+%Y-%m-%d %H:%M')   commit: $(git -C $HOME/QES rev-parse --short HEAD)"

# build QES-FitFloat with CUDA (make only recompiles what changed; output goes to build.txt)
if [ ! -d $FF_BUILD ]; then
  cmake -S $HOME/QES/QES-FitFloat -B $FF_BUILD -DENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=89 > $WORK/build.txt 2>&1
fi
make -C $FF_BUILD -j8 qesWinds >> $WORK/build.txt 2>&1 || { echo "build failed, see $WORK/build.txt"; exit 1; }

# command_for <version>: prints the command that runs that version on $CASE
command_for() {
  if [ "$1" = original ]; then
    echo "$BASELINE -q $CASE -s 3"
  elif [ "$1" = ff-build ]; then
    echo "$FITFLOAT -q $CASE -s 3"
  else
    echo "$FITFLOAT -q $CASE -s 3 -c $1"
  fi
}

# run_once "<command>": runs QES once; sets TIME (solver time, includes packing) and PACK (packing only, 0 if none)
run_once() {
  OUTPUT=$($1 2>/dev/null)
  TIME=$(echo "$OUTPUT" | awk '/Residual after/ {getline; print $3}')
  PACK=$(echo "$OUTPUT" | awk '/packing time/ {print $(NF-1)}')
  PACK=${PACK:-0}
}

# compare_fields <windsOut.nc> <windsOut.nc>: "identical", or the difference in u, v, w (python, numpy + netCDF4)
compare_fields() {
python3 - "$1" "$2" <<'PYEOF'
import sys
import numpy as np
from netCDF4 import Dataset
a, b = Dataset(sys.argv[1]), Dataset(sys.argv[2])
lines = []
for name in ("u", "v", "w"):
    x = np.asarray(a[name][:], dtype=np.float32)
    y = np.asarray(b[name][:], dtype=np.float32)
    if np.array_equal(x, y):
        continue
    d = np.abs(x.astype(np.float64) - y)
    lines.append(f"    {name}: max |diff| {np.nanmax(d):.2e} m/s, mean |diff| {np.nanmean(d):.2e} m/s, "
                 f"{np.count_nonzero(d > 0.01)} values off by more than 0.01 m/s, {np.count_nonzero(np.isnan(y))} NaN")
print("\n".join(lines) if lines else "    identical")
PYEOF
}

declare -A SUM MIN MAX PSUM REPEAT ERROR   # per version: times, packing time, 2nd-run check, error vs the original
cd $WORK

for CASE_NAME in okc slc; do
  if [ "$CASE_NAME" = okc ]; then
    CASE=$HOME/QES/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml; NTRIALS=5
  elif [ "$CASE_NAME" = slc ]; then
    CASE=$HOME/QES/QES-Public/tests/runTestsFiles/winds_SaltLakeCity.xml; NTRIALS=3
  fi
  echo; echo "=== $CASE_NAME ==="

  # warm-up: every version runs twice with its wind field saved (<name>_windsOut.nc, <name>_rerun_windsOut.nc),
  # and the files are compared right away, so only the original's file stays on disk (SLC files are 1.2 GB):
  #   2nd run vs 1st run  -> does this version give the same answer every time?
  #   1st run vs original -> the error
  ORIG=$WORK/${CASE_NAME}_original_windsOut.nc
  RAN=""
  for NAME in original ff-build $VARIANTS; do
    OUT=$WORK/${CASE_NAME}_$NAME
    if ! $(command_for $NAME) -o $OUT > $OUT.txt 2>&1 || ! $(command_for $NAME) -o ${OUT}_rerun > /dev/null 2>&1; then
      echo "$NAME failed, see $OUT.txt"
      if [ "$NAME" = original ]; then continue 2; fi   # nothing to compare against: skip this case
      continue
    fi
    RAN="$RAN $NAME"
    REPEAT[$NAME]=$(compare_fields ${OUT}_windsOut.nc ${OUT}_rerun_windsOut.nc)
    rm -f ${OUT}_rerun_windsOut.nc
    if [ "$NAME" != original ]; then
      ERROR[$NAME]=$(compare_fields $ORIG ${OUT}_windsOut.nc)
      rm -f ${OUT}_windsOut.nc
    fi
  done
  rm -f $ORIG

  # timed runs: every version in turn, so every version sees the same conditions
  for NAME in $RAN; do
    SUM[$NAME]=0; MIN[$NAME]=1000000; MAX[$NAME]=0; PSUM[$NAME]=0
  done
  for ((trial = 1; trial <= NTRIALS; trial++)); do
    for NAME in $RAN; do
      run_once "$(command_for $NAME)"
      SUM[$NAME]=$(echo "${SUM[$NAME]} + $TIME" | bc -l)
      PSUM[$NAME]=$(echo "${PSUM[$NAME]} + $PACK" | bc -l)
      if (( $(echo "$TIME < ${MIN[$NAME]}" | bc -l) )); then MIN[$NAME]=$TIME; fi
      if (( $(echo "$TIME > ${MAX[$NAME]}" | bc -l) )); then MAX[$NAME]=$TIME; fi
    done
  done

  echo "solver time (s), $NTRIALS alternating trials (FitFloat times include packing e..n once on the CPU + the GPU decode check):"
  printf "  %-11s %8s %8s %8s %8s %9s %10s\n" version min max avg packing speedup "w/o pack"
  ORIG_AVG=$(echo "${SUM[original]} / $NTRIALS" | bc -l)
  for NAME in $RAN; do
    AVG=$(echo "${SUM[$NAME]} / $NTRIALS" | bc -l)
    PACK=$(echo "${PSUM[$NAME]} / $NTRIALS" | bc -l)
    printf "  %-11s %8.4f %8.4f %8.4f %8.4f %8.2fx %9.2fx\n" $NAME ${MIN[$NAME]} ${MAX[$NAME]} $AVG $PACK \
      $(echo "$ORIG_AVG / $AVG" | bc -l) $(echo "$ORIG_AVG / ($AVG - $PACK)" | bc -l)
  done

  echo "what each variant stored (from QES's own output):"
  for V in $RAN; do
    case $V in original|ff-build) continue ;; esac
    echo "  $V:"
    grep "\[FitFloat\]" $WORK/${CASE_NAME}_$V.txt | grep -v "packing time" | sed 's/\[FitFloat\]\s*/    /'
  done

  echo "same answer every run? (each version's 2nd run vs its 1st; must be identical):"
  for NAME in $RAN; do
    echo "  $NAME:"
    echo "${REPEAT[$NAME]}"
  done

  echo "error in u, v, w vs the original (ff-build must be identical; 0.01 m/s = smallest wind change instruments can measure):"
  for NAME in $RAN; do
    if [ "$NAME" = original ]; then continue; fi
    echo "  $NAME:"
    echo "${ERROR[$NAME]}"
  done
done
