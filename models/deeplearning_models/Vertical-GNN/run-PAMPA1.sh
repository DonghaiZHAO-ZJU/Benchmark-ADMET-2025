#!/bin/bash

times=5
split_methods=("random" "scaffold" "Perimeter")
all_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss" "HIV_large" "PAMPA")
classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "HIV_large")
regression_tasks=("Caco2" "HalfLife" "VDss" "PAMPA" "PAMPA1")
selected_tasks=("PAMPA1")

for ((i=0; i<times; i++)); do
  seed=$((2024+i*10))
  for split_method in "${split_methods[@]}"; do
    for task in "${selected_tasks[@]}"; do
      echo "Current seed: $seed, current split method: $split_method, current task: $task"
      result_file="./result/Vertical-GNN_${task}_${split_method}_${seed}_all_result.csv"
      if test -f ${result_file} ; then
        echo "Result existed!"
      else
        var_name="${task}_${split_method}_${seed}"
        declare "$var_name"="some_value"
        if [[ " ${classification_tasks[@]} " =~ " ${task} " ]]; then
          if [[ "$task" == "oral_bioavailability" ]]; then
            command="python -u train.py --data_name ${task}_${split_method}_${seed} --task_type classification --patience 10 --learning_rate 0.00452976319043267"
          else
            command="python -u train.py --data_name ${task}_${split_method}_${seed} --task_type classification"
          fi
          echo "Executing command: $command"
          $command
        elif [[ " ${regression_tasks[@]} " =~ " ${task} " ]]; then
          command="python -u train.py --data_name ${task}_${split_method}_${seed} --task_type regression"
          echo "Executing command: $command"
          $command
        fi
      fi
    done
  done
done