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
TIMEFORMAT='whole run: %R s'           

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
    NCU_PICK="-k PotGlob -c 4"  # the fire kernel, the wind kernels are already covered by okc and slc
  elif [ "$CASE_NAME" = ffii ]; then
    CMD="$BUILD/qesFire/qesFire -q $HOME/QES/QES-Public/data/InputFiles/FFII.xml -s 3 -o $WORK/out/ffii -b"
    NCU_PICK="-k PotGlob -c 4"
  fi

  # 1. do a full run output stage times
  time $CMD > $WORK/${CASE_NAME}_qes.txt 2>&1
  if [ $? -ne 0 ]; then
    echo "QES failed, see $WORK/${CASE_NAME}_qes.txt"
    continue
  fi
  echo "stage times: seconds, count, stage"
  awk '/^\[/ {stage = $1" "$2" "$3} /elapsed time/ {total[stage] += $(NF-1); count[stage]++} END {for (s in total) printf "%9.3f %5d  %s\n", total[s], count[s], s}' $WORK/${CASE_NAME}_qes.txt | sort -rn

  # 2. GPU time per kernel: ncu times every launch (nsys sees no kernels under WSL)
  #    wind inputs only: the fire runs launch hundreds of thousands of kernels, too slow under ncu
  if [ "$CASE_NAME" = okc ] || [ "$CASE_NAME" = slc ]; then
    ncu --metrics gpu__time_duration.sum --clock-control none -f -o $WORK/${CASE_NAME}_time $CMD > /dev/null 2>&1
    echo "GPU kernels (ncu): % of kernel time, total ms, launches, kernel"
    ncu -i $WORK/${CASE_NAME}_time.ncu-rep | awk '/Context [0-9]/ {kernel = $1; sub(/\(.*/, "", kernel)} /gpu__time_duration.sum/ {t = $NF; gsub(/,/, "", t); u = $(NF-1); if (u ~ /^(ns|nsecond)$/) t /= 1e6; if (u ~ /^(us|usecond)$/) t /= 1e3; if (u ~ /^(s|second)$/) t *= 1e3; ms[kernel] += t; n[kernel]++; all += t} END {for (k in ms) printf "%6.1f %9.1f %6d  %s\n", 100*ms[k]/all, ms[k], n[k], k}' | sort -rn
  fi

  # 3. memory vs compute, averaged per kernel
  ncu --section SpeedOfLight $NCU_PICK -f -o $WORK/$CASE_NAME $CMD > /dev/null 2>&1
  echo "memory vs compute (ncu): DRAM %, SM %, launches measured, kernel"
  ncu -i $WORK/$CASE_NAME.ncu-rep | awk '/Context [0-9]/ {kernel = $1; sub(/\(.*/, "", kernel)} /DRAM Throughput/ {dram[kernel] += $NF; n[kernel]++} /Compute \(SM\) Throughput/ {sm[kernel] += $NF} END {for (k in n) printf "%6.1f %6.1f %6d  %s\n", dram[k]/n[k], sm[k]/n[k], n[k], k}' | sort -rn
done

# delete the big files, keep <case>_qes.txt
rm -f $WORK/out/*.nc $WORK/*.ncu-rep
