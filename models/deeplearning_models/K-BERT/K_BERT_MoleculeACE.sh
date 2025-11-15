#!/bin/bash

times=5
split_methods=("MoleculeACE")

for ((i=0; i<times; i++)); do
    seed=$((2024+i*10))
    for split_method in "${split_methods[@]}"; do
        echo "Current seed: $seed, current split method: $split_method"
        python practice.py --seed $seed --split_method $split_method --scaler StandardScaler
    done
done