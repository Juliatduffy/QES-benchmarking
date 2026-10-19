#!/bin/bash
# Compares the original QES GPU solver (QES-Public, -s 3) with FitFloat variants
# (QES-FitFloat, -s 3 -c <variant>) on OKC and SLC: solver time and the error in u, v, w.
#
# Run on the PC, from ~/QES (after profile_gpu.sh pc has built QES-Public):
#   bash wind_compare_fitfloat.sh                    all variants
#   bash wind_compare_fitfloat.sh ff-en16 ff-r16     only these
#
# Variants (what is stored as FitFloat; the math stays float):
#   ff-ctl      e..n in 32 bits (1+8+23): same values, no bytes saved -> the cost of unpacking alone
#   ff-en16     e..n in 16 bits (1+8+7)
#   ff-en4      e..n in  4 bits (1+3+0); falls back to float e..n if a value is out of range (OKC)
#   ff-r16      R in 16 bits (1+8+7)
#   ff-en16r16  e..n and R in 16 bits

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

echo "Original GPU solver vs FitFloat variants: $VARIANTS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)   date: $(date '+%Y-%m-%d %H:%M')"

# build QES-FitFloat with CUDA (make only recompiles what changed; output goes to build.txt)
if [ ! -d $FF_BUILD ]; then
  cmake -S $HOME/QES/QES-FitFloat -B $FF_BUILD -DENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=89 > $WORK/build.txt 2>&1
fi
make -C $FF_BUILD -j8 qesWinds >> $WORK/build.txt 2>&1 || { echo "build failed, see $WORK/build.txt"; exit 1; }

# run_once "<command>": runs QES once; sets TIME (solver time, includes packing) and PACK (packing only, 0 if none)
run_once() {
  OUTPUT=$($1 2>/dev/null)
  TIME=$(echo "$OUTPUT" | awk '/Residual after/ {getline; print $3}')
  PACK=$(echo "$OUTPUT" | awk '/packing time/ {print $(NF-1)}')
  PACK=${PACK:-0}
}

# compare_fields <original windsOut.nc> <variant windsOut.nc>: error in u, v, w (python, numpy + netCDF4)
compare_fields() {
python3 - "$1" "$2" <<'PYEOF'
import sys
import numpy as np
from netCDF4 import Dataset
a, b = Dataset(sys.argv[1]), Dataset(sys.argv[2])
for name in ("u", "v", "w"):
    x = np.asarray(a[name][:], dtype=np.float32)
    y = np.asarray(b[name][:], dtype=np.float32)
    if np.array_equal(x, y):
        print(f"    {name}: identical")
        continue
    d = np.abs(x.astype(np.float64) - y)
    print(f"    {name}: max |diff| {np.nanmax(d):.2e} m/s, mean |diff| {np.nanmean(d):.2e} m/s, "
          f"{np.count_nonzero(d > 0.01)} values off by more than 0.01 m/s, {np.count_nonzero(np.isnan(y))} NaN")
PYEOF
}

declare -A SUM MIN MAX PSUM   # per version: total, fastest and slowest solver time, total packing time
cd $WORK

for CASE_NAME in okc slc; do
  if [ "$CASE_NAME" = okc ]; then
    CASE=$HOME/QES/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml; NTRIALS=5
  elif [ "$CASE_NAME" = slc ]; then
    CASE=$HOME/QES/QES-Public/tests/runTestsFiles/winds_SaltLakeCity.xml; NTRIALS=3
  fi
  echo; echo "=== $CASE_NAME ==="

  # warm-up: one run of each version, saving its wind field (<name>_windsOut.nc) for the error check
  $BASELINE -q $CASE -s 3 -o $WORK/${CASE_NAME}_original > /dev/null 2>&1
  RAN=""
  for V in $VARIANTS; do
    if $FITFLOAT -q $CASE -s 3 -c $V -o $WORK/${CASE_NAME}_$V > $WORK/${CASE_NAME}_$V.txt 2>&1; then
      RAN="$RAN $V"
    else
      echo "$V failed, see $WORK/${CASE_NAME}_$V.txt"
    fi
  done

  # timed runs: original, variant 1, variant 2, ... in turn, so every version sees the same conditions
  for NAME in original $RAN; do
    SUM[$NAME]=0; MIN[$NAME]=1000000; MAX[$NAME]=0; PSUM[$NAME]=0
  done
  for ((trial = 1; trial <= NTRIALS; trial++)); do
    for NAME in original $RAN; do
      if [ "$NAME" = original ]; then
        run_once "$BASELINE -q $CASE -s 3"
      else
        run_once "$FITFLOAT -q $CASE -s 3 -c $NAME"
      fi
      SUM[$NAME]=$(echo "${SUM[$NAME]} + $TIME" | bc -l)
      PSUM[$NAME]=$(echo "${PSUM[$NAME]} + $PACK" | bc -l)
      if (( $(echo "$TIME < ${MIN[$NAME]}" | bc -l) )); then MIN[$NAME]=$TIME; fi
      if (( $(echo "$TIME > ${MAX[$NAME]}" | bc -l) )); then MAX[$NAME]=$TIME; fi
    done
  done

  echo "solver time (s), $NTRIALS alternating trials (FitFloat times include packing e..n once on the CPU):"
  printf "  %-11s %8s %8s %8s %8s %9s %10s\n" version min max avg packing speedup "w/o pack"
  ORIG_AVG=$(echo "${SUM[original]} / $NTRIALS" | bc -l)
  for NAME in original $RAN; do
    AVG=$(echo "${SUM[$NAME]} / $NTRIALS" | bc -l)
    PACK=$(echo "${PSUM[$NAME]} / $NTRIALS" | bc -l)
    printf "  %-11s %8.4f %8.4f %8.4f %8.4f %8.2fx %9.2fx\n" $NAME ${MIN[$NAME]} ${MAX[$NAME]} $AVG $PACK \
      $(echo "$ORIG_AVG / $AVG" | bc -l) $(echo "$ORIG_AVG / ($AVG - $PACK)" | bc -l)
  done

  echo "what each variant stored (from QES's own output):"
  for V in $RAN; do
    echo "  $V:"
    grep "\[FitFloat\]" $WORK/${CASE_NAME}_$V.txt | grep -v "packing time" | sed 's/\[FitFloat\]\s*/    /'
  done

  echo "error in u, v, w vs the original (0.01 m/s = smallest wind change instruments can measure):"
  for V in $RAN; do
    echo "  $V:"
    compare_fields $WORK/${CASE_NAME}_original_windsOut.nc $WORK/${CASE_NAME}_${V}_windsOut.nc
    rm -f $WORK/${CASE_NAME}_${V}_windsOut.nc
  done
  rm -f $WORK/${CASE_NAME}_original_windsOut.nc
done
