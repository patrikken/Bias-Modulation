#!/bin/bash

# Example: Run your script with 3 different seeds
 
configs=("generic" "1hot" "intersec" "dcp" "multitask" "bias_modulation") 
dataset_class_paths=("dataloaders.heart.Heart" "dataloaders.mimic.Mimic")
dataset_class_paths=("dataloaders.mimic.Mimic")
sensitive_feature_list=("['sex', 'age']")
sensitive_feature_list=("['sex', 'age']" "['sex']" "['age']")  
max_epoch=200
devices=4
accelerator="cpu"
strategy="ddp_find_unused_parameters_true"

if [[ $accelerator == "mps" ]]; then 
    strategy="auto"
fi

declare -A out_dir 

out_dir["heart"]="dataloaders.heart.Heart"
out_dir["mimic"]="dataloaders.mimic.Mimic"

dataset_names=("mimic" "heart")
 


configs=('bias_modulation')
for seed in 6 7 8 9 10; do 
    echo "Running with seed $seed" 
    for dataset in "${!out_dir[@]}"; do
        dataset_class=${out_dir[$dataset]}
        echo "  Dataset Class: $dataset_class"  
        for config in "${configs[@]}"; do  
            for sensitive_feature in "${sensitive_feature_list[@]}"; do
                echo "  Sensitive Feature: $sensitive_feature" 
                # Replace the following line with your actual command, e.g.:
                # python train.py --seed $seed
                if [[ $config == *"dcp"* ]]; then # Example condition to run a different script for certain configs
                    python train_decoupled.py --c $config --data.init_args.sensitive_attr "${sensitive_feature}" --trainer.devices $devices --seed_everything $seed --trainer.accelerator cpu --data.init_args.class_path $dataset_class --trainer.max_epochs $max_epoch
                else
                echo python train.py fit -c "configs/$config.yaml" --data.init_args.sensitive_attr "${sensitive_feature}" --trainer.devices $devices --seed_everything $seed --trainer.accelerator $accelerator --data.init_args.class_path $dataset_class --output_dir results/${dataset_names[$dataset]}/$config --trainer.strategy $strategy
                    python train.py fit -c "configs/$config.yaml" --data.init_args.sensitive_attr "${sensitive_feature}" --trainer.devices $devices --trainer.accelerator $accelerator --seed_everything $seed --trainer.accelerator $accelerator --data.init_args.class_path $dataset_class --trainer.max_epochs $max_epoch --output_dir results/${dataset_names[$dataset]}/$config --trainer.strategy $strategy
                    
                fi
            done
        done 
    done
    #
done

