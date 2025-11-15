#!/bin/bash

times=5
split_methods=("random" "scaffold" "Perimeter")
all_tasks=("Caco2")

for task in "${all_tasks[@]}"; do
    for split_method in "${split_methods[@]}"; do
        for ((i=0; i<times; i++)); do
            seed=$((2024+i*10))        
            echo "Current seed: $seed, current split method: $split_method, current task: $task"
            python create_data.py --data_name ${task}_${split_method}_${seed}
        done
    done
done
