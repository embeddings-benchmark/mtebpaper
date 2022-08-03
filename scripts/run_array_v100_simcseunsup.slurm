#!/bin/bash
#SBATCH --job-name=run-array-v100    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=10:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate muennighoffmtb

echo "START TIME: $(date)"

cd /gpfsscratch/rech/six/commun/commun/experiments/muennighoff/

NUM_TASKS_PER_JOB=1

TASK_START=$(expr $SLURM_ARRAY_TASK_ID \* $NUM_TASKS_PER_JOB )
TASK_END=$(expr $TASK_START + $NUM_TASKS_PER_JOB )

python mteb/run_array.py \
	--startid $TASK_START \
	--endid $TASK_END \
    --modelpath /gpfswork/rech/six/commun/models/princeton-nlp_unsup-simcse-bert-base-uncased
