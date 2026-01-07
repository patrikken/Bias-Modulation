#!/bin/bash

# Example: Run your script with 3 different seeds
module load python/3.12
source /home/joslin/scratch/Bias-Modulation/.env/bin/activate

methods=("1hot" "intersectional" "generic" "multitask" "decoupled")   # "bias_modulation" 
methods=("decoupled")   # "bias_modulation" 
sensitive_feature_list=("['sex', 'age']") # "['sex']" "['age']")  
#sensitive_attribs="['age']" #"['sex', 'age']" # "['sex']" "['age']"
dataset_name="sap"
for sensitive_attribs in "${sensitive_feature_list[@]}"; do
    for seed in {1..5}; do
    for method in "${methods[@]}"; do
        echo "Running method: $method with seed: $seed, for dataset: $dataset_name, sensitive attributes: ${sensitive_attribs}"
        python run.py --c configs/base.yaml --dataset_name "$dataset_name" --method_name "$method" --trainer.devices 1 --trainer.accelerator gpu --trainer.log_every_n_steps 1 --seed_everything "$seed" --data.init_args.sensitive_attr "${sensitive_attribs}"
        done
    done
done
