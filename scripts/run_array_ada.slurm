#!/bin/bash
#SBATCH --job-name=ada
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@cpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate muennighoffmtb

echo "START TIME: $(date)"

cd /gpfsscratch/rech/six/commun/commun/experiments/muennighoff/mteb

NUM_TASKS_PER_JOB=10

TASK_START=$(expr $SLURM_ARRAY_TASK_ID \* $NUM_TASKS_PER_JOB )
TASK_END=$(expr $TASK_START + $NUM_TASKS_PER_JOB )

python run_array_openai.py \
	--startid $TASK_START \
	--endid $TASK_END
