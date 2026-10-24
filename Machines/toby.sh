#!/bin/bash
# This file contains tools and settings necessary to run code on our pc at home

CMAKE_EXTRA=""
CUDA_ARCH="89"
THREADS_WIND_OKC="1 8 14"
THREADS_WIND_SLC="8 14"
THREADS_FIRE="8 14" # not including hyperthreads cuz idk how those work to be honest :|
NTRIALS_OKC=10
NTRIALS_SLC=5
NTRIALS_FIRE=5
NTRIALS_FIRE_QUICK=3


qes_env() {
    export OMP_PLACES=cores OMP_PROC_BIND=close
}