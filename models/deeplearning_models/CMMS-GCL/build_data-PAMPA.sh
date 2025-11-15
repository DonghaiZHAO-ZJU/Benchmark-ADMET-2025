#!/bin/bash

times=5
split_methods=("Perimeter")
all_tasks=("PAMPA1")

for ((i=0; i<times; i++)); do
    seed=$((2024+i*10))
    for split_method in "${split_methods[@]}"; do
        for task in "${all_tasks[@]}"; do
            echo "Current seed: $seed, current split method: $split_method, current task: $task"
            python create_data.py --data_name ${task}_${split_method}_${seed}
        done
    done
done
