#!/bin/bash

times=2
split_methods=("random" "scaffold" "Perimeter" "Maximum_Dissimilarity")
all_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss" "HIV_large")
classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "HIV_large")
regression_tasks=("Caco2" "HalfLife" "VDss")

for ((i=0; i<times; i++)); do
  seed=$((2024+i*10))
  for split_method in "${split_methods[@]}"; do
    for task in "${all_tasks[@]}"; do
      echo "Current seed: $seed, current split method: $split_method, current task: $task"
      var_name="${task}_${split_method}_${seed}"
      declare "$var_name"="some_value"
      if [[ " ${classification_tasks[@]} " =~ " ${task} " ]]; then
        chemprop_train --data_path /root/data1/admet_models_validation/chemprop-main/data/${task}_${split_method}_${seed}/${task}_${split_method}_${seed}_training.csv \
                       --separate_val_path /root/data1/admet_models_validation/chemprop-main/data/${task}_${split_method}_${seed}/${task}_${split_method}_${seed}_valid.csv \
                       --separate_test_path /root/data1/admet_models_validation/chemprop-main/data/${task}_${split_method}_${seed}/${task}_${split_method}_${seed}_test.csv \
                       --dataset_type classification \
                       --save_dir /root/data1/admet_models_validation/chemprop-main/result/${task}_${split_method}_${seed}_all_result \
                       --metric auc \
                       --extra_metrics prc-auc accuracy \
                       --seed 2024 \
                       --num_folds 5
      elif [[ " ${regression_tasks[@]} " =~ " ${task} " ]]; then
        chemprop_train --data_path /root/data1/admet_models_validation/chemprop-main/data/${task}_${split_method}_${seed}/${task}_${split_method}_${seed}_training.csv \
                       --separate_val_path /root/data1/admet_models_validation/chemprop-main/data/${task}_${split_method}_${seed}/${task}_${split_method}_${seed}_valid.csv \
                       --separate_test_path /root/data1/admet_models_validation/chemprop-main/data/${task}_${split_method}_${seed}/${task}_${split_method}_${seed}_test.csv \
                       --dataset_type regression \
                       --save_dir /root/data1/admet_models_validation/chemprop-main/result/${task}_${split_method}_${seed}_all_result \
                       --metric r2 \
                       --extra_metrics rmse mae \
                       --seed 2024 \
                       --num_folds 5        
        echo "Task $task is not defined as classification or regression task."
      fi
    done
  done
done