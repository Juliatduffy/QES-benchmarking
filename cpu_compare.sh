#!/bin/bash
# Times QES-Public against QES-CPU then checks that their output files are identical.
# Usage: 
#   bash cpu_compare.sh <wind|fire> <optimization> [cases]
# Examples:
#   bash cpu_compare.sh wind fuse  (runs okc and slc by default)
#   bash cpu_compare.sh fire combined2  (runs short by default)
#   bash cpu_compare.sh fire combined2 "tiny short"  (quotes needed for more than one case)
# To run on CHPC:
#   sbatch cpu_compare.sh wind fuse     

#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH -t 1:30:00
#SBATCH -o ./Testing/logs/cpu_compare.out

set -e # break on exceptions

QES_ROOT="$HOME/QES"
source "$QES_ROOT/helpers.sh"
select_machine

MODEL=$1            # wind or fire
OPT=$2              # optimization name passed to the optimized program with -c, ex: fuse
CASES=$3            # optional: which cases to run, e.g. "okc" or "okc slc"
KIND="cpu"
KEEP_OUTPUTS="no"   # "yes" keeps the .nc files in Testing/runs after the check
DEFAULT_FIRE_CASES="short"   # same case as the archived CHPC logs; long is too slow for the original CPU code

if [ -z "$MODEL" ] || [ -z "$OPT" ]
then
    echo "Usage: cpu_compare.sh <wind|fire> <optimization> [cases]"
    exit 1
fi

BASELINE_BUILD="cpu-original"
BASELINE_FLAGS="-s 1"
OPTIMIZED_BUILD="cpu-optimized"
OPTIMIZED_FLAGS="-s 1 -c $OPT"

run_all_cases
