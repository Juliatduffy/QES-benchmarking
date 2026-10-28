#!/bin/bash
# FitFloat study for QES-Plume: compares the original GPU particle model (qesPlumeGPU without -c)
# with the FitFloat variants (-c <variant>) that store the wind and turbulence fields bit-packed.
# Reports the time loop and the interpolation time, the packing cost, the run-to-run noise of every
# version (the GPU particle model is not deterministic), and the error of the time-averaged
# concentration field vs the original, to be read against that noise.
#
# Run from this directory (the plume-fitfloat worktree):
#   bash plume_compare_fitfloat.sh                      all variants, both domain sizes
#   bash plume_compare_fitfloat.sh ff-wt16 ff-wt24      only these variants
#   SIZES="large" bash plume_compare_fitfloat.sh        only the large domain
#   ARCH=80 EXTRA_CMAKE="-DNETCDF_CXX_DIR=..." bash plume_compare_fitfloat.sh    on CHPC granite (A100), after loadmodules_QES.sh
#
# Variants (what is packed; the math stays float; see src/plume/CUDA/Interpolation_FitFloat.h):
#   ff-ctl     winds and turbulence in 32 bits (1+8+23): same values -> the decode cost alone
#   ff-wt24    both in 24 bits (1+8+15)
#   ff-wt20    both in 20 bits (1+8+11)
#   ff-wt16    both in 16 bits (1+8+7)
#   ff-wt16e5  both in 16 bits (1+5+10)
#   ff-wt12    both in 12 bits (1+8+3)
#   ff-w16     winds 16 bits, turbulence 32 bits
#   ff-t16     winds 32 bits, turbulence 16 bits
# Domain sizes:
#   small  102 x 102 x 141 cells (the unit test's case; the 14 fields are 87 MB as float)
#   large  402 x 202 x 141 cells (the fields are 680 MB as float, far beyond the GPU's L2 cache)

VARIANTS="$*"
if [ -z "$VARIANTS" ]; then
  VARIANTS="ff-ctl ff-wt24 ff-wt20 ff-wt16 ff-wt16e5 ff-wt12 ff-w16 ff-t16"
fi
SIZES=${SIZES:-"small large"}
NTRIALS=${NTRIALS:-3}
STEPS=${STEPS:-1500}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARCH=${ARCH:-89}                       # CUDA architecture: 89 = RTX 4070 SUPER (PC), 80 = A100 (CHPC granite), like profile_gpu.sh
BUILD=$ROOT/build-gpu-sm$ARCH
EXE=$BUILD/qesPlume/qesPlumeGPU
WORK=$ROOT/Testing/plume_fitfloat
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

PY=python3
if [ -x "$HOME/.venvs/miniweather/bin/python" ]; then PY=$HOME/.venvs/miniweather/bin/python; fi
if ! $PY -c "import numpy, netCDF4" 2>/dev/null; then
  echo "the error check needs python's numpy and netCDF4"
  exit 1
fi

mkdir -p $WORK $ROOT/Testing/logs
LOG=$ROOT/Testing/logs/$(date +%Y-%m-%d_%H-%M-%S)_plume_fitfloat_compare.log
echo "Log: $LOG"
exec > >(tee "$LOG") 2>&1

echo "QES-Plume GPU: original vs FitFloat variants: $VARIANTS   sizes: $SIZES   trials: $NTRIALS   steps: $STEPS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)   date: $(date '+%Y-%m-%d %H:%M')   commit: $(git -C $ROOT rev-parse --short HEAD) ($(git -C $ROOT branch --show-current))"

# build (make only recompiles what changed)
if [ ! -d $BUILD ]; then
  cmake -S $ROOT/QES-FitFloat -B $BUILD -DENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DCMAKE_BUILD_TYPE=Release $EXTRA_CMAKE > $WORK/build.txt 2>&1
fi
make -C $BUILD -j8 qesPlumeGPU >> $WORK/build.txt 2>&1 || { echo "build failed, see $WORK/build.txt"; exit 1; }

# command_for <version> <size-args>
command_for() {
  if [ "$1" = original ]; then
    echo "$EXE -n $STEPS $2"
  else
    echo "$EXE -n $STEPS $2 -c $1"
  fi
}

# compare_fields <a_plumeOut.nc> <b_plumeOut.nc>: "identical" or the difference of the concentration field c
compare_fields() {
$PY - "$1" "$2" <<'PYEOF'
import sys
import numpy as np
from netCDF4 import Dataset
a, b = Dataset(sys.argv[1]), Dataset(sys.argv[2])
x = np.asarray(a["c"][:], dtype=np.float64).ravel()
y = np.asarray(b["c"][:], dtype=np.float64).ravel()
if np.array_equal(x, y):
    print("    identical")
else:
    d = np.abs(x - y)
    m = np.max(x) if np.max(x) > 0 else 1.0
    rel_rmse = np.sqrt(np.mean(d ** 2)) / m
    print(f"    c: rel RMSE {rel_rmse:.3e} of the peak, max |diff| {np.max(d) / m:.3e} of the peak, "
          f"{np.count_nonzero(d > 0.01 * m)} of {x.size} boxes off by > 1% of the peak, total count ratio {np.sum(y) / max(np.sum(x), 1e-30):.5f}")
PYEOF
}

# run_once "<command>": sets LOOP, INTERP, PACK, R2, ACTIVE from the summary line
run_once() {
  OUTPUT=$($1 2>&1)
  SUMMARY=$(echo "$OUTPUT" | grep "summary:")
  if [ -z "$SUMMARY" ]; then echo "run failed:"; echo "$OUTPUT" | tail -5; LOOP=nan; INTERP=nan; PACK=0; R2=nan; ACTIVE=0; return; fi
  LOOP=$(echo "$SUMMARY" | sed 's/.* loop=\([0-9.e+-]*\).*/\1/')
  INTERP=$(echo "$SUMMARY" | sed 's/.* interp=\([0-9.e+-]*\).*/\1/')
  PACK=$(echo "$SUMMARY" | sed 's/.* packing=\([0-9.e+-]*\).*/\1/')
  R2=$(echo "$SUMMARY" | sed 's/.* r2=\([0-9.e+-]*\).*/\1/')
  ACTIVE=$(echo "$SUMMARY" | sed 's/.* active=\([0-9]*\).*/\1/')
}

declare -A LSUM LMIN ISUM PSUM R2S NRUN REPEAT ERROR INFO
cd $WORK

for SIZE in $SIZES; do
  if [ "$SIZE" = small ]; then SIZE_ARGS="-x 102 -y 102 -z 141"; else SIZE_ARGS="-x 402 -y 202 -z 141"; fi
  echo; echo "=== $SIZE domain ($SIZE_ARGS) ==="

  # every version runs twice with its concentration saved: 2nd vs 1st run = same answer? 1st vs original = error
  ORIG=$WORK/${SIZE}_original_plumeOut.nc
  RAN=""
  for NAME in original $VARIANTS; do
    OUT=$WORK/${SIZE}_$NAME
    if ! $(command_for $NAME "$SIZE_ARGS") -o $OUT > $OUT.txt 2>&1 || ! $(command_for $NAME "$SIZE_ARGS") -o ${OUT}_rerun > /dev/null 2>&1; then
      echo "$NAME failed, see $OUT.txt"
      if [ "$NAME" = original ]; then continue 2; fi
      continue
    fi
    RAN="$RAN $NAME"
    REPEAT[$NAME]=$(compare_fields ${OUT}_plumeOut.nc ${OUT}_rerun_plumeOut.nc)
    rm -f ${OUT}_rerun_plumeOut.nc
    if [ "$NAME" != original ]; then
      ERROR[$NAME]=$(compare_fields $ORIG ${OUT}_plumeOut.nc)
    fi
    INFO[$NAME]=$(grep "\[FitFloat\]" $OUT.txt | grep -v "packing" | sed 's/\[FitFloat\]\s*/    /')
  done

  for NAME in $RAN; do LSUM[$NAME]=0; LMIN[$NAME]=1000000; ISUM[$NAME]=0; PSUM[$NAME]=0; R2S[$NAME]=0; NRUN[$NAME]=0; done
  for ((trial = 1; trial <= NTRIALS; trial++)); do
    for NAME in $RAN; do
      run_once "$(command_for $NAME "$SIZE_ARGS")"
      echo "  trial $trial $NAME: loop=$LOOP interp=$INTERP packing=$PACK r2=$R2 active=$ACTIVE"
      if ! [[ "$LOOP" =~ ^[0-9.]+$ && "$INTERP" =~ ^[0-9.]+$ && "$R2" =~ ^-?[0-9.]+$ ]]; then
        echo "  (trial $trial $NAME: could not parse the summary line, run skipped)"; continue
      fi
      NRUN[$NAME]=$(( ${NRUN[$NAME]:-0} + 1 ))
      LSUM[$NAME]=$(echo "${LSUM[$NAME]} + $LOOP" | bc -l)
      ISUM[$NAME]=$(echo "${ISUM[$NAME]} + $INTERP" | bc -l)
      PSUM[$NAME]=$(echo "${PSUM[$NAME]} + $PACK" | bc -l)
      R2S[$NAME]=$(echo "${R2S[$NAME]} + $R2" | bc -l)
      if (( $(echo "$LOOP < ${LMIN[$NAME]}" | bc -l) )); then LMIN[$NAME]=$LOOP; fi
    done
  done

  echo "time (s), $NTRIALS alternating trials, $STEPS steps, $ACTIVE active particles at the end; packing happens once before the loop and is not in the loop time:"
  printf "  %-10s %5s %9s %9s %9s %9s %9s %9s %8s\n" version runs loop_min loop_avg interp_avg speedup interp_x packing r2
  NO=${NRUN[original]:-0}
  OL=$(echo "${LSUM[original]} / $NO" | bc -l); OI=$(echo "${ISUM[original]} / $NO" | bc -l)
  for NAME in $RAN; do
    N=${NRUN[$NAME]:-0}
    if [ "$N" = 0 ]; then printf "  %-10s %5d  (no parsable runs)\n" $NAME 0; continue; fi
    L=$(echo "${LSUM[$NAME]} / $N" | bc -l); I=$(echo "${ISUM[$NAME]} / $N" | bc -l); P=$(echo "${PSUM[$NAME]} / $N" | bc -l); R=$(echo "${R2S[$NAME]} / $N" | bc -l)
    printf "  %-10s %5d %9.3f %9.3f %9.3f %8.2fx %8.2fx %9.3f %8.4f\n" $NAME $N ${LMIN[$NAME]} $L $I $(echo "$OL / $L" | bc -l) $(echo "$OI / $I" | bc -l) $P $R
  done

  echo "what each variant stored:"
  for NAME in $RAN; do
    [ "$NAME" = original ] && continue
    echo "  $NAME:"; echo "${INFO[$NAME]}"
  done
  echo "run-to-run noise (2nd run vs 1st; the GPU particle model is not deterministic: particle compaction uses atomics, so particles draw different random numbers):"
  for NAME in $RAN; do echo "  $NAME:"; echo "${REPEAT[$NAME]}"; done
  echo "error of the time-averaged concentration vs the original (compare with the run-to-run noise above; ff-ctl stores the same values):"
  for NAME in $RAN; do
    [ "$NAME" = original ] && continue
    echo "  $NAME:"; echo "${ERROR[$NAME]}"
  done
done
