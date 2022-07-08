#!/bin/bash
#SBATCH --job-name=run-mteb-5B8    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --reservation=hug
#SBATCH --constraint=a100
#SBATCH --partition=gpu_p5

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate muennighoffmtb

cd /gpfsscratch/rech/six/commun/commun/experiments/muennighoff/
# use SCRATCH for building as it's much faster
python mteb/run_5B8.py
