#!/bin/bash

#SBATCH --account=def-ebrahimi-ab_gpu 
#SBATCH --job-name=BiasModulation
#SBATCH --output=.slurm/logs/run_%j.out 
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100:1
#SBATCH --time=01:00:00 
#SBATCH --array=0-5

module load python/3.12
source /home/joslin/scratch/Bias-Modulation/.env

echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"

echo "Runing the command:"
echo "python hyperparameter_search.py --c configs/base.yaml --dataset_name "$DATASET_NAME" --method_name "$METHOD_NAME" --seed_everything $SLURM_ARRAY_TASK_ID --trainer.devices 1 --trainer.accelerator cuda --trainer.log_every_n_steps 1"

python hyperparameter_search.py --c configs/base.yaml --dataset_name "$DATASET_NAME" --method_name "$METHOD_NAME" --seed_everything $SLURM_ARRAY_TASK_ID --trainer.devices 1 --trainer.accelerator gpu --trainer.log_every_n_steps 1