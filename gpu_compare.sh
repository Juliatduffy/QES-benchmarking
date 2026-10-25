#!/bin/bash
# Times QES-Public (gpu-original) against QES-GPU (gpu-optimized), both with the
# global-memory GPU wind solver (-s 3), then checks that their output files are identical.
# Usage:
#   bash gpu_compare.sh <wind|fire> [cases]
# Examples:
#   bash gpu_compare.sh wind  (runs okc and slc by default)
#   bash gpu_compare.sh fire  (runs long by default, ~11 min on toby)
#   bash gpu_compare.sh fire tiny  (quick correctness check)
# To run on CHPC:
#   sbatch gpu_compare.sh fire


#SBATCH -M granite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-students-grn
#SBATCH --account=cs6230
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -t 1:30:00
#SBATCH -o ./Testing/logs/gpu_compare.out

# TODO: check on CHPC which partition / qos has GPUs for account cs6230 and fix the lines above
# also find a way to evaluate correctess with a certain error tolerance for lossy optimizations

set -e # break on exceptions

QES_ROOT="$HOME/QES"
source "$QES_ROOT/helpers.sh"
select_machine

MODEL=$1            # wind or fire
CASES=$2            # (optional) which cases to run ex: "okc slc" or "tiny long"
KIND="gpu"
KEEP_OUTPUTS="no"   # option to keep the .nc files in Testing/runs, otherwise they are deleted
DEFAULT_FIRE_CASES="long"   # short/tiny finish in 1-3 s on the GPU, too fast to time reliably
OPT=""              # no -c in QES-GPU yet

if [ -z "$MODEL" ]
then
    echo "Usage: gpu_compare.sh <wind|fire> [cases]"
    exit 1
fi

BASELINE_BUILD="gpu-original"
BASELINE_FLAGS="-s 3"
OPTIMIZED_BUILD="gpu-optimized"
OPTIMIZED_FLAGS="-s 3" # TODO implement -c

run_all_cases
