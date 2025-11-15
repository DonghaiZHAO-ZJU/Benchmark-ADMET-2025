#!/bin/bash

times=5
split_methods=("random" "scaffold" "Perimeter" "Maximum_Dissimilarity")
all_tasks=("PAMPA1")
classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "HIV_large")
regression_tasks=("Caco2" "HalfLife" "VDss" "PAMPA" "PAMPA1")


for task in "${all_tasks[@]}"; do
  echo "current task: $task"
  if [[ " ${classification_tasks[@]} " =~ " ${task} " ]]; then
    python -u finetune_class2.py  --task_name ${task} --data_name none \
                                  --data_path ./data/raw_data --processed_data_path ./data/processed_data \
                                  --group_data_path ./data/group \
                                  --compound_encoder_config model_configs/geognn_l8.json  --model_config model_configs/down_mlp2.json \
                                  --init_model ./pretrain_models-chemrl_gem/class.pdparams  --model_dir ./model \
                                  --task data
  elif [[ " ${regression_tasks[@]} " =~ " ${task} " ]]; then
    python -u finetune_regr2.py --task_name ${task} --data_name none \
                                --data_path ./data/raw_data --processed_data_path ./data/processed_data \
                                --group_data_path ./data/group \
                                --compound_encoder_config model_configs/geognn_l8.json  --model_config model_configs/down_mlp2.json \
                                --init_model ./pretrain_models-chemrl_gem/regr.pdparams  --model_dir ./model \
                                --task data  
  fi
done