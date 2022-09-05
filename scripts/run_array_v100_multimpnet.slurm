#!/bin/bash
#SBATCH --job-name=run-array-v100-multimpnet    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@gpu
#SBATCH -C v100-32g

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate muennighoffmtb

echo "START TIME: $(date)"

cd /gpfsscratch/rech/six/commun/commun/experiments/muennighoff/

NUM_TASKS_PER_JOB=1

TASK_START=$(expr $SLURM_ARRAY_TASK_ID \* $NUM_TASKS_PER_JOB )
TASK_END=$(expr $TASK_START + $NUM_TASKS_PER_JOB )

python mteb/run_array.py \
    --taskname BUCC \
	--startid $TASK_START \
	--endid $TASK_END \
    --modelpath /gpfswork/rech/six/commun/models/sentence-transformers_paraphrase-multilingual-mpnet-base-v2
