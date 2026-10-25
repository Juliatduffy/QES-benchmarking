#!/bin/bash
# This file contains tools and settings necessary to run code on CHPC machines
# for selecting cuda architecture on chpc machines:
# A100 = 80
# H100 = 90
# L40S = 89
# check with "sinfo -o "%P %G" | grep -i gpu"

CMAKE_EXTRA="-DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include"
CUDA_ARCH="80"
THREADS_WIND_OKC="1 16 64"
THREADS_WIND_SLC="16 64"
THREADS_FIRE="16 64"
NTRIALS_OKC=10
NTRIALS_SLC=5
NTRIALS_FIRE=5
NTRIALS_FIRE_QUICK=3
NTRIALS_FIRE_LONG=5

qes_env() {
    source "${QES_ROOT}/QES-Public/CHPC/loadmodules_QES.sh" # silence with > /dev/null 2>&1
    module load nccmp/1.9.1.0 # silence with > /dev/null 2>&1
    export OMP_PLACES=cores OMP_PROC_BIND=close
}