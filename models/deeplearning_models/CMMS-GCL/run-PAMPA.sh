#!/bin/bash

times=5
split_methods=("Perimeter")
all_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss" "HIV_large" "PAMPA")
classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability")
regression_tasks=("Caco2" "HalfLife" "VDss" "PAMPA1")
selected_tasks=("PAMPA1")

for ((i=0; i<times; i++)); do
  seed=$((2024+i*10))
  for split_method in "${split_methods[@]}"; do
    for task in "${selected_tasks[@]}"; do
      echo "Current seed: $seed, current split method: $split_method, current task: $task"
      var_name="${task}_${split_method}_${seed}"
      declare "$var_name"="some_value"
      if [[ " ${classification_tasks[@]} " =~ " ${task} " ]]; then
        python Training.py --data_name ${task}_${split_method}_${seed} --task_type classification
      elif [[ " ${regression_tasks[@]} " =~ " ${task} " ]]; then
        python Training.py --data_name ${task}_${split_method}_${seed} --task_type regression  
      fi
    done
  done
done