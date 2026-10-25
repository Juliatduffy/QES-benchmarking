#!/bin/bash
# Usage:
#   bash gpu_profile.sh <nsys|ncu> <wind|fire> <case> [kernel] [build] [optimization]
# Examples:
#   bash gpu_profile.sh nsys fire long                                 
#   bash gpu_profile.sh ncu fire short                                 
#   bash gpu_profile.sh ncu fire short PotGlob                         
#   bash gpu_profile.sh nsys fire long SOR gpu-optimized fuse          

set -e # break on exceptions

QES_ROOT="$HOME/QES"
source "$QES_ROOT/helpers.sh"
select_machine

TOOL=$1            # nsys or ncu 
MODEL=$2            # wind or fire
CASE_NAME=$3
KERNEL=${4:-SOR}                # kernel name to profile (if using ncu) default is sor global solver
BUILD_NAME=${5:-gpu-original}   # which build to profile default is qes gpu
OPT=${6:-original}              # which -c version to run 

if [ "$OPT" != "original" ] && [ "$BUILD_NAME" != "gpu-optimized" ]; then
    echo "ERROR: -c $OPT only exists in the gpu-optimized build"
    exit 1
fi

pick_case
pick_binary

OUT_DIR="$QES_ROOT/Testing/profile/$(date +%Y-%m-%d_%H-%M-%S)_${MACHINE}_${TOOL}_${MODEL}_${CASE_NAME}_${OPT}"
mkdir -p "$OUT_DIR"

if [ "$TOOL" = "nsys" ]; then
    PROFILER="nsys profile -o $OUT_DIR/report --trace=cuda"
else
    PROFILER="ncu --set detailed -k regex:$KERNEL -s 20 -c 5 -o $OUT_DIR/report"
fi

if [ "$MODEL" = "wind" ]; then
    RUN_FLAGS="-s 3 -w"
else
    RUN_FLAGS="-s 3 -b"
fi

if [ "$OPT" != "original" ]; then
    RUN_FLAGS="$RUN_FLAGS -c $OPT"   # QES-Public does not know -c, so only pass it for the optimized versions
fi

OUTPUT_BASENAME="$OUT_DIR/run"

run_once

echo "$RUN_OUTPUT" > "$OUT_DIR/stdout.txt" 
rm -f "$OUT_DIR"/run_*.nc
# write the profiler's key tables to summary.txt and show them
if [ "$TOOL" = "nsys" ]; then
    # time per kernel (:base = short kernel names), then time and size of memory copies
    nsys stats --force-export=true --format table \
        --report cuda_gpu_kern_sum:base --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum \
        "$OUT_DIR/report.nsys-rep" | grep -v "^Generating\|^Processing" > "$OUT_DIR/summary.txt"
    rm -f "$OUT_DIR/report.sqlite"   # cache made by nsys stats, recreated whenever needed
else
    # min / max / average over the profiled launches of each kernel
    ncu --import "$OUT_DIR/report.ncu-rep" --print-kernel-base function --print-summary per-kernel \
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy > "$OUT_DIR/summary.txt"
fi
cat "$OUT_DIR/summary.txt"
echo "Profile saved in: $OUT_DIR"
