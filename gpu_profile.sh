#!/bin/bash

set -e # break on exceptions

QES_ROOT="$HOME/QES"
source "$QES_ROOT/helpers.sh"
select_machine

TOOL=$1            # nsys or ncu 
MODEL=$2            # wind or fire
CASE_NAME=$3
KERNEL=${4:-SOR}                # kernel name to profile (if using ncu) default is sor global solver
BUILD_NAME=${5:-gpu-original}   # which build to profile default is qes gpu

pick_case
pick_binary

OUT_DIR="$QES_ROOT/Testing/profile/$(date +%Y-%m-%d_%H-%M-%S)_${MACHINE}_${TOOL}_${MODEL}_${CASE_NAME}"
mkdir -p "$OUT_DIR"

if [ "$TOOL" = "nsys" ]; then
    PROFILER="nsys profile -o $OUT_DIR/report --trace=cuda --stats=true"
else
    PROFILER="ncu --set detailed -k regex:$KERNEL -s 20 -c 5 -o $OUT_DIR/report"
fi

if [ "$MODEL" = "wind" ]; then
    RUN_FLAGS="-s 3 -w"
else
    RUN_FLAGS="-s 3 -b"
fi

OUTPUT_BASENAME="$OUT_DIR/run"

run_once

echo "$RUN_OUTPUT" > "$OUT_DIR/stdout.txt" # can add --stats=true
rm -f "$OUT_DIR"/run_*.nc