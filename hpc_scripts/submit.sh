#!/bin/bash

datasets=("mimic" "heart")
datasets=("heart")
methods=("bias_modulation" "1hot" "intersectional" "generic" "multitask")

for dataset in "${datasets[@]}"; do
  for method in "${methods[@]}"; do
    sbatch --export=DATASET_NAME="$dataset",METHOD_NAME="$method" hpc_scripts/hpc.sh
  done
done