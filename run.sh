#!/bin/bash

# Example: Run your script with 3 different seeds

configs=("configs/acs_pums/acs_pum_income.yaml"  "configs/adult/adult_age.yaml" "configs/adult/adult_race.yaml" "configs/adult/adult_sex.yaml")
configs=("configs/heart/heart_generic.yaml" "configs/heart/heart_1hot.yaml" "configs/heart/heart_intersec.yaml" "configs/heart/heart_dcp.yaml" "configs/heart/heart_multitask.yaml") 

sensitive_feature_list=("['sex', 'age']")
sensitive_feature_list=("['sex']" "['age']" "['sex', 'age']")  
configs=("configs/heart/heart_bias_modulation.yaml ")
for seed in 6 7 8 9 10; do 
    echo "Running with seed $seed"
    for config in "${configs[@]}"; do  
        for sensitive_feature in "${sensitive_feature_list[@]}"; do
            echo "  Sensitive Feature: $sensitive_feature" 
            # Replace the following line with your actual command, e.g.:
            # python train.py --seed $seed
            if [[ $config == *"dcp.yaml"* ]]; then # Example condition to run a different script for certain configs
                python train_decoupled.py --c $config --data.init_args.sensitive_attr "${sensitive_feature}" --trainer.devices 1 --seed_everything $seed --trainer.accelerator cpu
            else
            echo python train.py fit -c $config --data.init_args.sensitive_attr "${sensitive_feature}" --trainer.devices 1 --seed_everything $seed --trainer.accelerator cpu
                python train.py fit -c $config --data.init_args.sensitive_attr "${sensitive_feature}" --trainer.devices 1 --seed_everything $seed --trainer.accelerator cpu
            fi
        done
    done 
    #
done

