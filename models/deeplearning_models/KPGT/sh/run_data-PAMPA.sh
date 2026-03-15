#!/bin/bash
#SBATCH -J practice
#SBATCH -o practice-%j.log
#SBATCH -e practice-%j.err
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
##SBATCH --gres=gpu:1
##SBATCH -w gpu19  # If you want to specify a computing node, you can write its name here and remove the first #

all_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss")
selected_tasks=("PAMPA")

for task in "${selected_tasks[@]}"; do
  python preprocess_downstream_dataset.py --data_path ../datasets/ --dataset $task
done