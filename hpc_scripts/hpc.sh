#!/bin/bash

#SBATCH --account=def-ebrahimi-ab 
#SBATCH --job-name=BiasModulation
#SBATCH --output=.slurm/logs/run_%j_%a.out 
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:2
#SBATCH --time=04:30:00  
#SBATCH --array=1-1

module load python/3.12
source /home/joslin/scratch/Bias-Modulation/.env/bin/activate

echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"

SEED=$SLURM_ARRAY_TASK_ID
echo "Runing the command:"
sensitive_attribs="${1}"
echo "python hyperparameter_search.py --c configs/base.yaml --dataset_name "$DATASET_NAME" --method_name "$METHOD_NAME" --seed_everything $SEED --trainer.devices 1 --trainer.accelerator gpu --trainer.log_every_n_steps 1" --data.init_args.sensitive_attr "${sensitive_attribs}"

CUDA_VISIBLE_DEVICES=0 python hyperparameter_search.py --c configs/base.yaml --dataset_name "$DATASET_NAME" --method_name "$METHOD_NAME" --seed_everything $SEED --trainer.devices 2 --trainer.accelerator gpu --trainer.log_every_n_steps 1 --data.init_args.sensitive_attr "${sensitive_attribs}"