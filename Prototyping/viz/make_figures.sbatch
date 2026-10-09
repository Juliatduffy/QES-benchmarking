#!/bin/bash
#SBATCH -M granite
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-shortbatch-grn
#SBATCH --account=cs6230
#SBATCH -t 0:10:00
#SBATCH -J qes-figures
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1417379/QES/Testing/figures/make_figures_%j.log
# Render all figures headless on a compute node.  Submit from anywhere:  sbatch Testing/viz/make_figures.sbatch
VIZ=/uufs/chpc.utah.edu/common/home/u1417379/QES/Testing/viz
echo "node: $SLURMD_NODENAME  job: $SLURM_JOB_ID  $(date)"
module --force purge; module load chpc; module load paraview/6.0.1
for script in speed_map.py views_2d.py streamlines_3d.py; do
    echo "=== $script"
    pvbatch --force-offscreen-rendering $VIZ/$script 2>&1 | grep -v "bad X server connection"
done
