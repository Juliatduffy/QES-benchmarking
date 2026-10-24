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
   echo "Host name: ${hostname}"
   echo "Date: ${date}"
}
