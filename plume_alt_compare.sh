#!/bin/bash
# Plume field storage study: FitFloat vs other ways to shrink (or re-lay-out) the 14 read-only fields that the
# GPU particle model (qesPlumeGPU, uniform-flow case) interpolates every step.
#
#   bash plume_alt_compare.sh                      everything (defaults below), ~1-1.5 h on an A100
#   SIZES="large" NTRIALS=2 bash plume_alt_compare.sh          quicker
#   VARIANTS="h16 aos32" bash plume_alt_compare.sh             only these (the original always runs)
#
# Environment (all optional):
#   EXE       path to an already built qesPlumeGPU (otherwise this script builds QES-FitFloat in build-gpu-sm$ARCH)
#   ARCH      CUDA architecture for the build (default: detected with nvidia-smi, e.g. 80 for an A100)
#   EXTRA_CMAKE  extra cmake arguments for the build (e.g. -DNETCDF_CXX_DIR=...)
#   SIZES     domain presets, run in this order (default "large small xlarge"):
#               small  102 x 102 x 141 (fields 83 MB)      large  402 x 202 x 141 (648 MB)
#               xlarge 802 x 402 x 141 (2.6 GB)            dense  large with 30 000 new particles per step
#   VARIANTS  storage variants (default below); see QES-FitFloat/src/plume/CUDA/Interpolation_FitFloat.h
#   NTRIALS   timed runs per version per size, alternating (default 3); every timed run also saves its
#             concentration, so the trials double as the accuracy samples
#   STEPS     time steps per run (default 1500)
#   NVERIFY   positions for the deterministic interpolation check (default 1000000)
#   NCU       1 = profile the interpolation kernels with Nsight Compute on the large domain, 0 = skip,
#             auto (default) = only if ncu is on the PATH
#   WORK      output directory (default Testing/plume_alt/<date>_<host>)
#
# Writes: $WORK/plume_alt.log (the summary to read), per-run outputs and *.txt next to it.

set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
HOST=$(hostname -s)
VARIANTS=${VARIANTS:-"ff-ctl ff-wt16 ff-wt16e5 f32 nonut h16 bf16 q16 aos32 aos16"}
SIZES=${SIZES:-"large small xlarge"}
NTRIALS=${NTRIALS:-3}
STEPS=${STEPS:-1500}
NVERIFY=${NVERIFY:-1000000}
NCU=${NCU:-auto}
NCU_STEPS=${NCU_STEPS:-600}
WORK=${WORK:-$ROOT/Testing/plume_alt/${STAMP}_${HOST}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PY=${PY:-python3}

mkdir -p "$WORK"
LOG=$WORK/plume_alt.log
exec > >(tee "$LOG") 2>&1

GPU=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "plume field storage study   host: $HOST   GPU: $GPU   date: $(date '+%Y-%m-%d %H:%M')"
echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) ($(git -C "$ROOT" branch --show-current 2>/dev/null)), uncommitted changes: $(git -C "$ROOT" status --porcelain 2>/dev/null | grep -c -E 'Interpolation_FitFloat|qesPlumeGPU')"
echo "variants: $VARIANTS   sizes: $SIZES   trials: $NTRIALS   steps: $STEPS   verify positions: $NVERIFY   work: $WORK"

# ---- build (unless EXE is given) ----
if [ -z "${EXE:-}" ]; then
  ARCH=${ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')}
  [ -z "$ARCH" ] && { echo "could not detect the GPU architecture; set ARCH (e.g. ARCH=80)"; exit 1; }
  BUILD=${BUILD:-$ROOT/build-gpu-sm$ARCH}
  if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "^CMAKE_HOME_DIRECTORY:INTERNAL=$ROOT/QES-FitFloat\$" "$BUILD/CMakeCache.txt"; then
    echo "$BUILD was configured for another source tree; building in $ROOT/build-plumealt-sm$ARCH instead"
    BUILD=$ROOT/build-plumealt-sm$ARCH
  fi
  EXE=$BUILD/qesPlume/qesPlumeGPU
  echo "building qesPlumeGPU for sm_$ARCH in $BUILD (log: $WORK/build.txt)"
  if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    cmake -S "$ROOT/QES-FitFloat" -B "$BUILD" -DENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DCMAKE_BUILD_TYPE=Release ${EXTRA_CMAKE:-} > "$WORK/build.txt" 2>&1 \
      || { echo "cmake failed, see $WORK/build.txt"; tail -20 "$WORK/build.txt"; exit 1; }
  fi
  NJ=$(nproc 2>/dev/null || echo 8); [ "$NJ" -gt 32 ] && NJ=32
  make -C "$BUILD" -j$NJ qesPlumeGPU >> "$WORK/build.txt" 2>&1 || { echo "build failed, see $WORK/build.txt"; grep -n error "$WORK/build.txt" | head -20; exit 1; }
fi
[ -x "$EXE" ] || { echo "no executable at $EXE"; exit 1; }
for V in $VARIANTS; do
  "$EXE" -h 2>&1 | grep -q -- "$V" || { echo "this qesPlumeGPU does not know variant '$V' (old build?)"; exit 1; }
done
HAVE_NUMPY=1
$PY -c "import numpy" 2>/dev/null || { HAVE_NUMPY=0; echo "WARNING: $PY has no numpy: the concentration accuracy tables are skipped (timing and the verify check still run)"; }

size_args() {
  case $1 in
    small) echo "-x 102 -y 102 -z 141" ;;
    large) echo "-x 402 -y 202 -z 141" ;;
    xlarge) echo "-x 802 -y 402 -z 141" ;;
    dense) echo "-x 402 -y 202 -z 141 -N 30000" ;;
    *) echo "unknown size $1" >&2; exit 1 ;;
  esac
}
cmd_for() { # <version> <size args>
  if [ "$1" = original ]; then echo "$EXE $2 -n $STEPS"; else echo "$EXE $2 -n $STEPS -c $1"; fi
}
field() { echo "$1" | sed -n "s/.* $2=\([-0-9.e+]*\).*/\1/p"; }

# ---- 1. deterministic check: what each storage does to the values (once, on the first size) ----
FIRST=${SIZES%% *}
echo
echo "=== 1. storage statistics and deterministic interpolation check ($FIRST domain, $NVERIFY random positions) ==="
echo "    [Storage] = per field, what a read returns vs the float value (relative errors over nonzero values; bias = mean signed relative error)"
echo "    [verify]  = per interpolated output, variant kernel vs original kernel at the same positions"
for V in $VARIANTS; do
  echo "--- $V"
  if "$EXE" $(size_args $FIRST) -c $V -V $NVERIFY > "$WORK/verify_$V.txt" 2>&1; then
    grep -E '^\[(Storage|verify|FitFloat)\]' "$WORK/verify_$V.txt" | grep -v -E 'packing (winds|time)|GPU decode mismatches: 0' | sed 's/^\[\([A-Za-z]*\)\]\t */  [\1] /'
  else
    echo "  verify run FAILED:"; tail -5 "$WORK/verify_$V.txt"
  fi
done

# ---- 2. timing + accuracy samples per size ----
declare -A LSUM LMIN ISUM N
for SIZE in $SIZES; do
  SA=$(size_args $SIZE)
  echo
  echo "=== 2. $SIZE domain ($SA): $NTRIALS alternating trials of $STEPS steps ==="
  for NAME in original $VARIANTS; do LSUM[$NAME]=0; ISUM[$NAME]=0; LMIN[$NAME]=1000000000; N[$NAME]=0; done
  # untimed warm-up (GPU clocks, driver, file system), so the first timed version is not penalised
  $(cmd_for original "$SA" | sed "s/-n $STEPS/-n 300/") > $WORK/${SIZE}_warmup.txt 2>&1
  for ((t = 1; t <= NTRIALS; t++)); do
    for NAME in original $VARIANTS; do
      OUT=$WORK/${SIZE}_${NAME}_t$t
      $(cmd_for $NAME "$SA") -o $OUT > $OUT.txt 2>&1
      S=$(grep "summary:" $OUT.txt)
      if [ -z "$S" ]; then echo "  trial $t $NAME FAILED:"; tail -3 $OUT.txt; continue; fi
      L=$(field "$S" loop); I=$(field "$S" interp); A=$(field "$S" active); P=$(field "$S" packing); R=$(field "$S" r2)
      echo "  trial $t $(printf '%-10s' $NAME) loop $L s   interp $I s   packing $P s   active $A   r2 $R"
      if ! [[ "$L" =~ ^[0-9.]+$ && "$I" =~ ^[0-9.]+$ ]]; then echo "    (unparsable or negative time, trial not counted)"; continue; fi
      rm -f ${OUT}_plumeOut.nc
      N[$NAME]=$((N[$NAME] + 1))
      LSUM[$NAME]=$(echo "${LSUM[$NAME]} + $L" | bc -l)
      ISUM[$NAME]=$(echo "${ISUM[$NAME]} + $I" | bc -l)
      if (($(echo "$L < ${LMIN[$NAME]}" | bc -l))); then LMIN[$NAME]=$L; fi
    done
  done
  echo "  time (s), averages over the counted trials; speedup = original avg / version avg"
  printf "    %-10s %4s %9s %9s %10s %8s %9s %10s\n" version runs loop_min loop_avg interp_avg speedup interp_x MB_fields
  OL=$(echo "${LSUM[original]} / ${N[original]}" | bc -l 2>/dev/null)
  OI=$(echo "${ISUM[original]} / ${N[original]}" | bc -l 2>/dev/null)
  for NAME in original $VARIANTS; do
    if [ "${N[$NAME]}" = 0 ]; then printf "    %-10s %4d  (no runs)\n" $NAME 0; continue; fi
    LA=$(echo "${LSUM[$NAME]} / ${N[$NAME]}" | bc -l); IA=$(echo "${ISUM[$NAME]} / ${N[$NAME]}" | bc -l)
    MB=$(grep -h "device bytes for the 14 fields" $WORK/${SIZE}_${NAME}_t1.txt 2>/dev/null | sed -n 's/.*-> \([0-9.]*\) MB packed.*/\1/p')
    printf "    %-10s %4d %9.3f %9.3f %10.3f %7.2fx %8.2fx %10s\n" $NAME ${N[$NAME]} ${LMIN[$NAME]} $LA $IA $(echo "$OL / $LA" | bc -l) $(echo "$OI / $IA" | bc -l) "${MB:-float}"
  done
  if [ $HAVE_NUMPY = 1 ]; then $PY "$ROOT/plume_alt_analyze.py" "$WORK" "$SIZE" original $VARIANTS; fi
done

# ---- 3. Nsight Compute: what the interpolation kernels do in memory (large domain, one late step) ----
if [ "$NCU" = auto ]; then if command -v ncu > /dev/null; then NCU=1; else NCU=0; echo; echo "(ncu not found: profiling skipped)"; fi; fi
if [ "$NCU" = 1 ]; then
  echo
  echo "=== 3. Nsight Compute, large domain, the 3 interpolation kernels of step $NCU_STEPS (caches not flushed, clocks not locked) ==="
  M=gpu__time_duration.sum,dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_op_read_hit_rate.pct,l1tex__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread
  SKIP=$((3 * (NCU_STEPS - 1)))
  UNITS=""; ncu --help 2>&1 | grep -q -- --print-units && UNITS="--print-units base"
  echo "  $(ncu --version 2>&1 | tail -1)"
  CSVS=""
  for NAME in original ff-ctl ff-wt16 h16 bf16 nonut aos32 aos16; do
    [ "$NAME" != original ] && ! echo " $VARIANTS " | grep -q " $NAME " && continue
    C=""; [ "$NAME" != original ] && C="-c $NAME"
    ncu --cache-control none --clock-control none -k regex:interpolate --launch-skip $SKIP --launch-count 3 \
      --metrics $M --page raw --csv $UNITS \
      $EXE $(size_args large) -n $NCU_STEPS $C > $WORK/ncu_$NAME.csv 2> $WORK/ncu_$NAME.err
    CSVS="$CSVS $WORK/ncu_$NAME.csv"
  done
  $PY "$ROOT/plume_alt_ncu.py" $CSVS
fi
echo
echo "done $(date '+%Y-%m-%d %H:%M'); log: $LOG"
