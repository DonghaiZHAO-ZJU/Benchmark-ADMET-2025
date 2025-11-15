#!/bin/bash

times=5
split_methods=("Perimeter")
all_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss" 'constrained_logP')
classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability")
regression_tasks=("Caco2" "HalfLife" "VDss" "PAMPA" "PAMPA1" 'constrained_logP')
selected_tasks=('PAMPA1')

sample_ratios=(0.1 0.2 0.4 0.6 0.8)  # 定义样本比例数组
# sample_ratios=(0.1 0.2 0.4)  # 定义样本比例数组

for sample_ratio in "${sample_ratios[@]}"; do
  for ((i=0; i<times; i++)); do
    seed=$((2024+i*10))
    for split_method in "${split_methods[@]}"; do
      for task in "${selected_tasks[@]}"; do
        echo "Current seed: $seed, current split method: $split_method, current task: $task"
        var_name="${split_method}_${seed}_${sample_ratio}"
        result_file="../result/KPGT_${task}_${split_method}_${seed}_${sample_ratio}_all_result.csv"
        if test -f ${result_file} ; then
          echo "Result existed!"
        else
          if [[ " ${classification_tasks[@]} " =~ " ${task} " ]]; then
            python finetune_new-mini.py --config base --model_path ../models/pretrained/base/base.pth --dataset $task --data_path ../datasets/ --dataset_type classification --metric rocauc prauc acc --split "$var_name" --weight_decay 0 --dropout 0 --lr 3e-5 --seed 2024 
          elif [[ " ${regression_tasks[@]} " =~ " ${task} " ]]; then
            python finetune_new-mini.py --config base --model_path ../models/pretrained/base/base.pth --dataset $task --data_path ../datasets/ --dataset_type regression --metric r2 rmse mae --split "$var_name" --weight_decay 0 --dropout 0 --lr 3e-5 --seed 2024
          else
            echo "Task $task is not defined as classification or regression task."
          fi
        fi
      done
    done
  done
done