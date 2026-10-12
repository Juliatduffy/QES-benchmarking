#!/bin/bash -x
#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shortbatch-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=16
#SBATCH -t 0:10:00
#SBATCH -o ./Testing/logs/build.out
echo "*** Assigned Granite Node: " $SLURMD_NODENAME
cd ~/QES
source QES-Public/CHPC/loadmodules_QES.sh

FLAGS="-DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include -DENABLE_OPENMP=ON -DHAS_CUDA_SUPPORT=OFF"

cmake -S QES-Public -B build-public $FLAGS
make -C build-public -j16 qesWinds

cmake -S QES-CPU -B build-cpu $FLAGS
make -C build-cpu -j16 qesWinds

