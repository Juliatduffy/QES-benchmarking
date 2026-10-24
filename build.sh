#!/bin/bash
# Build script to build QES code.

#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn          
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=64
#SBATCH --exclusive 
#SBATCH -t 0:30:00
#SBATCH -o ./Testing/logs/build.out

set -e # break on exceptions

QES_ROOT="$HOME/QES"
source "$QES_ROOT/helpers.sh"
select_machine

CPU_FLAGS="-DENABLE_OPENMP=ON -DHAS_CUDA_SUPPORT=OFF $CMAKE_EXTRA"
GPU_FLAGS="-DENABLE_OPENMP=ON -DHAS_CUDA_SUPPORT=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH $CMAKE_EXTRA"
SOURCE=""
FLAGS=""

for NAME in cpu-original gpu-original cpu-optimized gpu-optimized; do
    if [ "$NAME" = "cpu-original" ] 
    then
       SOURCE="QES-Public"
       FLAGS=$CPU_FLAGS
    elif [ "$NAME" = "gpu-original" ] 
    then
        SOURCE="QES-Public"
        FLAGS=$GPU_FLAGS
    elif [ "$NAME" = "cpu-optimized" ] 
    then
        SOURCE="QES-CPU"
        FLAGS=$CPU_FLAGS
    elif [ "$NAME" = "gpu-optimized" ] 
    then
        SOURCE="QES-GPU"
        FLAGS=$GPU_FLAGS
    fi

    cmake -S $QES_ROOT/$SOURCE -B $QES_ROOT/build/$NAME $FLAGS
    make -C $QES_ROOT/build/$NAME -j"$(nproc)" qesWinds qesFire
done