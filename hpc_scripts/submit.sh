#!/bin/bash

datasets=("mimic" "heart")
datasets=("mimic" "heart" "sap" "mortality")
methods=("bias_modulation" "1hot" "intersectional" "generic" "multitask" "decoupled") 
#methods=("multitask")
#methods=("decoupled")
#sensitive_feature_list=("['sex', 'age']" "['sex']" "['age']")  
#sensitive_feature_list=("['age']")   

script_path="hpc_scripts/hpc.sh"

if [ "$1" == "hyperparam" ]; then
  script_path="hpc_scripts/hpc_with_hyperaram.sh"
fi

#sensitive_attribs="['sex']" #"['sex', 'age']" #"['age']"

sensitive_feature_list=("['sex', 'age']" "['age']" "['sex']")  
#sensitive_feature_list=("['sex']")  
echo "Using script: $script_path"
for dataset in "${datasets[@]}"; do
  for method in "${methods[@]}"; do
    for sensitive_attribs in "${sensitive_feature_list[@]}"; do
      sbatch --export=DATASET_NAME="$dataset",METHOD_NAME="$method",SENSITIVE_FEATURES="${sensitive_attribs}" $script_path "${sensitive_attribs}"
    done
  done
done