#!/bin/bash
#SBATCH -J practice
#SBATCH -o practice-%j.log
#SBATCH -e practice-%j.err
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -w gpu20  # If you want to specify a computing node, you can write its name here and remove the first #

times=5
split_methods=("random" "scaffold" "Perimeter" "Maximum_Dissimilarity")

for ((i=0; i<times; i++)); do
    seed=$((2024+i*10))
    for split_method in "${split_methods[@]}"; do
        echo "Current seed: $seed, current split method: $split_method"
        python practice.py --seed $seed --split_method $split_method --scaler StandardScaler
    done
done

python build_dataset_for_tasks.py

for ((i=0; i<times; i++)); do
    seed=$((2024+i*10))
    for split_method in "${split_methods[@]}"; do
        echo "Current seed: $seed, current split method: $split_method"
        python practice-PAMPA.py --seed $seed --split_method $split_method --scaler RobustScaler
    done
done