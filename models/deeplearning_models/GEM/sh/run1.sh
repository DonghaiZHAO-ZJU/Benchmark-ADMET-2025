#!/bin/bash

times=2
split_methods=("random" "scaffold" "Perimeter" "Maximum_Dissimilarity")
all_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss" "HIV_large")
classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "HIV_large")
regression_tasks=("Caco2" "HalfLife" "VDss")

for ((i=0; i<times; i++)); do
  seed=$((2044+i*10))
  for split_method in "${split_methods[@]}"; do
    for task in "${all_tasks[@]}"; do
      echo "Current seed: $seed, current split method: $split_method, current task: $task"
      result_file="./result/GEM_${task}_${split_method}_${seed}_all_result.csv"
      if test -f ${result_file} ; then
        echo "Result existed!"
      else
        echo "Result no existed!"
        if [[ " ${classification_tasks[@]} " =~ " ${task} " ]]; then
          python finetune_class2.py --task_name ${task} --data_name ${task}_${split_method}_${seed} \
                                    --data_path ./data/raw_data --processed_data_path ./data/processed_data  --group_data_path ./data/group \
                                    --compound_encoder_config model_configs/geognn_l8.json  --model_config model_configs/down_mlp2.json  --init_model ./pretrain_models-chemrl_gem/class.pdparams  \
                                    --model_dir ./model
        elif [[ " ${regression_tasks[@]} " =~ " ${task} " ]]; then
          python finetune_regr2.py --task_name ${task} --data_name ${task}_${split_method}_${seed} \
                                  --data_path ./data/raw_data --processed_data_path ./data/processed_data --group_data_path ./data/group \
                                  --compound_encoder_config model_configs/geognn_l8.json  --model_config model_configs/down_mlp2.json  --init_model ./pretrain_models-chemrl_gem/regr.pdparams \
                                  --model_dir ./model 
        fi
      fi
    done
  done
done