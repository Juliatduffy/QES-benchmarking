#!/bin/bash
#SBATCH -M notchpeak
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --qos=notchpeak-gpu-guest
#SBATCH --gres=gpu:a100:1,nsight:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o ./Testing/logs/gpu_profile.out

# run QES 3x per input file, each with a different profiling tool
# 1. timing test to see which files consume most runtime
# 2. ncu timing every kernel launch to see which kernels take the longest (wind inputs only)
# 3. ncu to see where each kernel is stuck

# time_by_file <QES output> <whole run seconds>: adds up QES's own "elapsed time" lines
# and names the source file behind each one; whatever QES doesn't time is the last row
time_by_file() {
  awk -v whole="$2" '
    /^\[/ { stage = $0 }                      # the last QES message before a time is its label
    /elapsed time/ {
      if      (stage ~ /Current time/)          next   # one whole fire step = the rows below added up
      if      (stage ~ /Residual after/)        what = "GPU      Solver_GPU_GlobalMemory.cu (wind solve)"
      else if (stage ~ /GPU POTENTIAL/)         what = "GPU+CPU  PotentialGlobal.cu (plume winds)"
      else if (stage ~ /Computing level set/)   what = "CPU      LevelSetNB.cpp (spread rate + heat)"
      else if (stage ~ /dt =/)                  what = "CPU      Move.cpp (move the fire front)"
      else if (stage ~ /LS init/)               what = "GPU      LSinitGlob.cu (level set setup)"
      else if (stage ~ /Applying Wind Profile/) what = "CPU      WINDSGeneralData.cpp (wind profile)"
      else if (stage ~ /Applying/)              what = "CPU      WINDSGeneralData.cpp (building parameterizations)"
      else if (stage ~ /Defining solid walls/)  what = "CPU      WINDSGeneralData.cpp (solid walls)"
      else if (stage ~ /Sorting buildings/)     what = "CPU      WINDSGeneralData.cpp (buildings from shapefile)"
      else if (stage ~ /ESRIShapefile/)         what = "CPU      ESRIShapefile.cpp (read shapefile)"
      else                                      what = "?        " stage
      total[what] += $(NF-1); count[what]++; timed += $(NF-1)
    }
    END {
      for (w in total) printf "%9.3f %6.1f%% %5d  %s\n", total[w], 100 * total[w] / whole, count[w], w
      printf "%9.3f %6.1f%% %5s  %s\n", whole - timed, 100 * (whole - timed) / whole, "-", "CPU      untimed: startup, reading input, writing output"
    }' "$1" | sort -rn
}

MACHINE=$1 # chpc or toby's pc
if [ "$MACHINE" = chpc ]; then
  source $HOME/QES/QES-Public/CHPC/loadmodules_QES.sh
  ARCH=80
  WORK=/scratch/general/vast/$USER/gpu_profile
elif [ "$MACHINE" = pc ]; then
  ARCH=89
  EXTRA_CMAKE=""
  WORK=$HOME/gpu_profile
else
  echo "usage: profile_gpu.sh chpc|pc"
  exit 1
fi

BUILD=$HOME/QES/build-gpu-public-sm$ARCH
export OMP_NUM_THREADS=8                

# everything printed after this point goes into the log
mkdir -p $HOME/QES/Testing/logs $WORK/run $WORK/out
LOG=$HOME/QES/Testing/logs/$(date +%Y-%m-%d_%H-%M-%S)_gpu_profile_$MACHINE.log
echo "Log: $LOG"
exec > "$LOG" 2>&1

echo "QES-Public GPU profile (-s 3)"
echo "machine: $MACHINE   job: ${SLURM_JOB_ID:-none}   date: $(date '+%Y-%m-%d %H:%M')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [ ! -x $BUILD/qesFire/qesFire ]; then
  cmake -S $HOME/QES/QES-Public -B $BUILD -DENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH $EXTRA_CMAKE > $WORK/build.txt 2>&1
  make -C $BUILD -j8 qesWinds qesFire >> $WORK/build.txt 2>&1 || { echo "build failed, see $WORK/build.txt"; exit 1; }
fi

ln -sfn $HOME/QES/QES-Public/data $WORK/data
ln -sfn $HOME/QES/QES-Public/data/FireFiles $WORK/FireFiles
cd $WORK/run

for CASE_NAME in okc slc fire ffii; do
  echo; echo "=== $CASE_NAME ==="

  # which command to profile, and which kernel launches ncu measures
  if [ "$CASE_NAME" = okc ]; then
    CMD="$BUILD/qesWinds/qesWinds -q $HOME/QES/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml -s 3"
    NCU_PICK="-c 10"            # the first 10 launches cover every solver kernel except finalVelocityGlobal
  elif [ "$CASE_NAME" = slc ]; then
    CMD="$BUILD/qesWinds/qesWinds -q $HOME/QES/QES-Public/tests/runTestsFiles/winds_SaltLakeCity.xml -s 3"
    NCU_PICK="-c 10"
  elif [ "$CASE_NAME" = fire ]; then
    CMD="$BUILD/qesFire/qesFire -q $HOME/QES/QES-Public/data/InputFiles/fire.xml -s 3 -o $WORK/out/fire -b"
    NCU_PICK="-k PotGlob -c 4"  # the fire kernel (the wind kernels get a second ncu run below)
  elif [ "$CASE_NAME" = ffii ]; then
    CMD="$BUILD/qesFire/qesFire -q $HOME/QES/QES-Public/data/InputFiles/FFII.xml -s 3 -o $WORK/out/ffii -b"
    NCU_PICK="-k PotGlob -c 4"
  fi

  # 1. do a full run, then time per source file
  START=$(date +%s.%N)
  if ! $CMD > $WORK/${CASE_NAME}_qes.txt 2>&1; then
    echo "QES failed, see $WORK/${CASE_NAME}_qes.txt"
    continue
  fi
  WHOLE=$(awk -v start=$START -v end=$(date +%s.%N) 'BEGIN {printf "%.3f", end - start}')
  echo "whole run: $WHOLE s"
  echo "time by file: seconds, % of run, count, where, file (what)"
  time_by_file $WORK/${CASE_NAME}_qes.txt $WHOLE

  # 2. GPU time per kernel: ncu times every launch (nsys sees no kernels under WSL)
  #    wind inputs only: the fire runs launch hundreds of thousands of kernels, too slow under ncu
  if [ "$CASE_NAME" = okc ] || [ "$CASE_NAME" = slc ]; then
    ncu --metrics gpu__time_duration.sum --clock-control none -f -o $WORK/${CASE_NAME}_time $CMD > /dev/null 2>&1
    echo "GPU kernels (ncu): % of kernel time, total ms, launches, kernel"
    ncu -i $WORK/${CASE_NAME}_time.ncu-rep | awk '/Context [0-9]/ {kernel = $1; sub(/\(.*/, "", kernel)} /gpu__time_duration.sum/ {t = $NF; gsub(/,/, "", t); u = $(NF-1); if (u ~ /^(ns|nsecond)$/) t /= 1e6; if (u ~ /^(us|usecond)$/) t /= 1e3; if (u ~ /^(s|second)$/) t *= 1e3; ms[kernel] += t; n[kernel]++; all += t} END {for (k in ms) printf "%6.1f %9.1f %6d  %s\n", 100*ms[k]/all, ms[k], n[k], k}' | sort -rn
  fi

  # 3. memory vs compute, averaged per kernel
  ncu --section SpeedOfLight $NCU_PICK -f -o $WORK/$CASE_NAME $CMD > /dev/null 2>&1
  REPORTS=$WORK/$CASE_NAME.ncu-rep
  if [ "$CASE_NAME" = fire ] || [ "$CASE_NAME" = ffii ]; then
    # also the wind solver's kernels: their names all end in "Global" (the fire grid is a different size from okc and slc)
    ncu --section SpeedOfLight -k regex:Global -c 10 -f -o $WORK/${CASE_NAME}_wind $CMD > /dev/null 2>&1
    REPORTS="$REPORTS $WORK/${CASE_NAME}_wind.ncu-rep"
  fi
  echo "memory vs compute (ncu): DRAM %, SM %, launches measured, kernel"
  for REPORT in $REPORTS; do ncu -i $REPORT; done | awk '/Context [0-9]/ {kernel = $1; sub(/\(.*/, "", kernel)} /DRAM Throughput/ {dram[kernel] += $NF; n[kernel]++} /Compute \(SM\) Throughput/ {sm[kernel] += $NF} END {for (k in n) printf "%6.1f %6.1f %6d  %s\n", dram[k]/n[k], sm[k]/n[k], n[k], k}' | sort -rn
done

# delete the big files, keep <case>_qes.txt
rm -f $WORK/out/*.nc $WORK/*.ncu-rep
