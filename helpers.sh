#!/bin/bash
# helper functions for the other bash scripts
# note that in bash all variables are global by default, so this script uses
# variables defined by cpu_compare.sh, gpu_compare.sh, etc.

select_machine() {
  if [ -d /uufs/chpc.utah.edu ]; then
    MACHINE="chpc"
  elif [ "$(hostname)" = "DESKTOP-VJ703QM" ]; then
    MACHINE="toby"
  else
    echo "ERROR: unknown machine $(hostname); add it to select_machine in helpers.sh"; exit 1
  fi
  source "$QES_ROOT/Machines/$MACHINE.sh"
  qes_env
}

print_header(){
   echo "Machine: ${MACHINE}"
   echo "Host name: $(hostname)"
   echo "Date: $(date)"
}

# Turns a case alias into the input file, the folder to run it from, the
# thread list, and trial count for that case.
pick_case() {
  if [ "$CASE_NAME" = "okc" ]
  then
    CASE_FILE="$QES_ROOT/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml"
    RUN_DIR="$QES_ROOT"
    THREADS="$THREADS_WIND_OKC"
    NTRIALS="$NTRIALS_OKC"
  elif [ "$CASE_NAME" = "slc" ]
  then
    CASE_FILE="$QES_ROOT/QES-Public/tests/runTestsFiles/winds_SaltLakeCity.xml"
    RUN_DIR="$QES_ROOT"
    THREADS="$THREADS_WIND_SLC"
    NTRIALS="$NTRIALS_SLC"
  elif [ "$CASE_NAME" = "short" ]
  then
    CASE_FILE="$QES_ROOT/Testing/inputs/fire_short.xml"
    RUN_DIR="$QES_ROOT/QES-Public/qesFire" 
    THREADS="$THREADS_FIRE"
    NTRIALS="$NTRIALS_FIRE"
  elif [ "$CASE_NAME" = "tiny" ]
  then
    CASE_FILE="$QES_ROOT/Testing/inputs/fire_tiny.xml"
    RUN_DIR="$QES_ROOT/QES-Public/qesFire" 
    THREADS="$THREADS_FIRE"
    NTRIALS="$NTRIALS_FIRE_QUICK"
  elif [ "$CASE_NAME" = "long" ]
  then 
    CASE_FILE="$QES_ROOT/Testing/inputs/fire_long.xml"
    RUN_DIR="$QES_ROOT/QES-Public/qesFire" 
    THREADS="$THREADS_FIRE"
    NTRIALS="$NTRIALS_FIRE_LONG"
  else
    echo "ERROR: unknown case '$CASE_NAME' (known cases: okc slc short tiny long)"
    exit 1
  fi
}

# Build the path to a compiled QES program and make sure it exists
pick_binary() {
  if [ "$MODEL" = "wind" ]
  then
    BINARY="$QES_ROOT/build/$BUILD_NAME/qesWinds/qesWinds"
  else
    BINARY="$QES_ROOT/build/$BUILD_NAME/qesFire/qesFire"
  fi
  if [ ! -x "$BINARY" ]
  then
    echo "ERROR: $BINARY not found, run build.sh first"
    exit 1
  fi
}

# Run one program once and keep everything it printed for the correctness check
run_once() {
  cd "$RUN_DIR"
  if ! RUN_OUTPUT=$("$BINARY" -q "$CASE_FILE" -o "$OUTPUT_BASENAME" $RUN_FLAGS 2>&1)
  then
    echo "$RUN_OUTPUT"
    echo "ERROR: $BINARY failed on $CASE_FILE"
    exit 1
  fi
  cd "$QES_ROOT"
}

# Parse the timings out of one run's printed output
parse_times() {
  if [ "$MODEL" = "wind" ]
  then
    TIME_SOL=$(echo "$RUN_OUTPUT" | awk '/Residual after/ { getline; sum += $3 } END { print sum }')
    TIME_POT=0
    TIME_FIRE=0
    if [ -z "$TIME_SOL" ]
    then
      echo "$RUN_OUTPUT"
      echo "ERROR: no 'Residual after' line in the output above, cannot time this run"
      exit 1
    fi
  else
    TIMES=$(echo "$RUN_OUTPUT" | awk '
      /POTENTIAL/           { in_pot = 1 }
      /Residual after/      { in_sol = 1; next }
      /Fire step completed/ { in_fire = 1 }
      /elapsed time/ {
        if (in_pot)       { pot  += $3; in_pot = 0 }
        else if (in_sol)  { sol  += $3; in_sol = 0 }
        else if (in_fire) { fire += $3; in_fire = 0 }
      }
      END { printf "%f %f %f\n", fire, pot, sol }')
    TIME_FIRE=$(echo "$TIMES" | awk '{ print $1 }')
    TIME_POT=$(echo "$TIMES" | awk '{ print $2 }')
    TIME_SOL=$(echo "$TIMES" | awk '{ print $3 }')
    if [ "$TIME_FIRE" = "0.000000" ]
    then
      echo "$RUN_OUTPUT"
      echo "ERROR: no 'Fire step completed' line in the output above, cannot time this run"
      exit 1
    fi
  fi
}

# Compare two output files value by value with nccmp to make sure they match
compare_outputs() {
  if [ "$MODEL" = "wind" ]
  then
    BASE_FILE="${BASE_OUT}_windsWk.nc"
    OPT_FILE="${OPT_OUT}_windsWk.nc"
    NCCMP_VARS="-v u,v,w"
  else
    BASE_FILE="${BASE_OUT}_fireOutput.nc"
    OPT_FILE="${OPT_OUT}_fireOutput.nc"
    NCCMP_VARS=""
  fi
  if nccmp -d -S -f -q $NCCMP_VARS "$BASE_FILE" "$OPT_FILE"
  then
    COMPARE_RESULT="IDENTICAL"
  else
    COMPARE_RESULT="DIFFERENT"
  fi
}

# Run the baseline program once, write its output files to BASE_OUT
run_baseline() {
  BINARY="$BASELINE_BIN"
  RUN_FLAGS="$BASELINE_FLAGS $OUTPUT_FLAGS"
  OUTPUT_BASENAME="$BASE_OUT"
  run_once
}

# Run the optimized program once write its output files to OPT_OUT
run_optimized() {
  BINARY="$OPTIMIZED_BIN"
  RUN_FLAGS="$OPTIMIZED_FLAGS $OUTPUT_FLAGS"
  OUTPUT_BASENAME="$OPT_OUT"
  run_once
}

# Time the baseline program against the optimized one on a single case, then
# check that their outputs are identical for the correctness check
run_comparison() {
  pick_case
  if [ "$MODEL" = "wind" ]
  then
    OUTPUT_FLAGS="-w" 
  else
    OUTPUT_FLAGS="-b" 
  fi

  BUILD_NAME="$BASELINE_BUILD"
  pick_binary
  BASELINE_BIN="$BINARY"
  BUILD_NAME="$OPTIMIZED_BUILD"
  pick_binary
  OPTIMIZED_BIN="$BINARY"

  mkdir -p "$QES_ROOT/Testing/runs"
  BASE_OUT="$QES_ROOT/Testing/runs/${KIND}_${CASE_NAME}_baseline"
  OPT_OUT="$QES_ROOT/Testing/runs/${KIND}_${CASE_NAME}_optimized"

  # warm up run
  run_baseline
  run_optimized

  for NT in $THREADS
  do
    export OMP_NUM_THREADS=$NT
    BASE_SUM_FIRE=0
    BASE_SUM_POT=0
    BASE_SUM_SOL=0
    OPT_SUM_FIRE=0
    OPT_SUM_POT=0
    OPT_SUM_SOL=0

    # alternate baseline / optimized so both see the same machine conditions
    for TRIAL in $(seq 1 "$NTRIALS")
    do
      run_baseline
      parse_times
      BASE_SUM_FIRE=$(echo "$BASE_SUM_FIRE + $TIME_FIRE" | bc -l)
      BASE_SUM_POT=$(echo "$BASE_SUM_POT + $TIME_POT" | bc -l)
      BASE_SUM_SOL=$(echo "$BASE_SUM_SOL + $TIME_SOL" | bc -l)

      run_optimized
      parse_times
      OPT_SUM_FIRE=$(echo "$OPT_SUM_FIRE + $TIME_FIRE" | bc -l)
      OPT_SUM_POT=$(echo "$OPT_SUM_POT + $TIME_POT" | bc -l)
      OPT_SUM_SOL=$(echo "$OPT_SUM_SOL + $TIME_SOL" | bc -l)
    done

    echo
    echo "$NT threads, average of $NTRIALS alternating trials (seconds):"
    if [ "$MODEL" = "wind" ]
    then
      printf "  %-16s wind solver %8.3f\n" "$BASELINE_BUILD"  $(echo "$BASE_SUM_SOL / $NTRIALS" | bc -l)
      printf "  %-16s wind solver %8.3f\n" "$OPTIMIZED_BUILD" $(echo "$OPT_SUM_SOL / $NTRIALS" | bc -l)
      printf "  Speedup (wind solver): %.2fx\n" $(echo "$BASE_SUM_SOL / $OPT_SUM_SOL" | bc -l)
    else
      printf "  %-16s fire loop %8.2f   potential %8.2f   wind solver %8.2f\n" "$BASELINE_BUILD" \
        $(echo "$BASE_SUM_FIRE / $NTRIALS" | bc -l) $(echo "$BASE_SUM_POT / $NTRIALS" | bc -l) $(echo "$BASE_SUM_SOL / $NTRIALS" | bc -l)
      printf "  %-16s fire loop %8.2f   potential %8.2f   wind solver %8.2f\n" "$OPTIMIZED_BUILD" \
        $(echo "$OPT_SUM_FIRE / $NTRIALS" | bc -l) $(echo "$OPT_SUM_POT / $NTRIALS" | bc -l) $(echo "$OPT_SUM_SOL / $NTRIALS" | bc -l)
      printf "  Speedup (fire loop): %.2fx\n" $(echo "$BASE_SUM_FIRE / $OPT_SUM_FIRE" | bc -l)
    fi
  done

  # correctness check
  compare_outputs
  echo
  echo "Correctness ($OPTIMIZED_BUILD output vs $BASELINE_BUILD output): $COMPARE_RESULT"

  if [ "$KEEP_OUTPUTS" = "no" ]
  then
    rm -f "${BASE_OUT}"_* "${OPT_OUT}"_*
  fi
}

# shared on cpu and gpu runs, pick the default cases, start the log file, 
# print the header, then compare each case.
run_all_cases() {
  if [ -z "$CASES" ]
  then
    if [ "$MODEL" = "wind" ]
    then
      CASES="okc slc"
    else
      CASES="$DEFAULT_FIRE_CASES"
    fi
  fi

  if [ -z "$OPT" ]
  then
    LOG_NAME="${MACHINE}_${KIND}_${MODEL}"
  else
    LOG_NAME="${MACHINE}_${KIND}_${MODEL}_${OPT}"
  fi
  mkdir -p "$QES_ROOT/Testing/logs"
  LOG="$QES_ROOT/Testing/logs/$(date +%Y-%m-%d_%H-%M-%S)_${LOG_NAME}.log"
  echo "Summary will be written to: $LOG"
  exec > "$LOG" 2>&1  # start logging

  print_header
  for CASE_NAME in $CASES
  do
    run_comparison
  done
}
