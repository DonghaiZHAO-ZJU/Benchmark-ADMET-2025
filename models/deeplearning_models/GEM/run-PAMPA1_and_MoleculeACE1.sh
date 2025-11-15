#!/bin/bash

# times=5
# split_methods=("random" "scaffold" "Perimeter")
# all_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss" "HIV_large" "PAMPA")
# classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "HIV_large")
# regression_tasks=("Caco2" "HalfLife" "VDss" "PAMPA" "PAMPA1")
# selected_tasks=("PAMPA1")

# for ((i=0; i<times; i++)); do
#   seed=$((2024+i*10))
#   for split_method in "${split_methods[@]}"; do
#     for task in "${selected_tasks[@]}"; do
#       echo "Current seed: $seed, current split method: $split_method, current task: $task"
#       result_file="./result/GEM_${task}_${split_method}_${seed}_all_result.csv"
#       if test -f ${result_file} ; then
#         echo "Result existed!"
#       else
#         echo "Result no existed!"
#         if [[ " ${classification_tasks[@]} " =~ " ${task} " ]]; then
#           python finetune_class2.py --task_name ${task} --data_name ${task}_${split_method}_${seed} \
#                                     --data_path ./data/raw_data --processed_data_path ./data/processed_data  --group_data_path ./data/group \
#                                     --compound_encoder_config model_configs/geognn_l8.json  --model_config model_configs/down_mlp2.json  --init_model ./pretrain_models-chemrl_gem/class.pdparams  \
#                                     --model_dir ./model
#         elif [[ " ${regression_tasks[@]} " =~ " ${task} " ]]; then
#           python finetune_regr2-PAMPA.py --task_name ${task} --data_name ${task}_${split_method}_${seed} \
#                                   --data_path ./data/raw_data --processed_data_path ./data/processed_data --group_data_path ./data/group \
#                                   --compound_encoder_config model_configs/geognn_l8.json  --model_config model_configs/down_mlp2.json  --init_model ./pretrain_models-chemrl_gem/regr.pdparams \
#                                   --model_dir ./model 
#         fi
#       fi
#     done
#   done
# done

times=5
split_methods=("MoleculeACE")
all_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss" "HIV_large" "PAMPA")
classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "HIV_large")
regression_tasks=("Caco2" "HalfLife" "VDss" "PAMPA" "PAMPA1" "CHEMBL1862_Ki" "CHEMBL1871_Ki" "CHEMBL2034_Ki" "CHEMBL2047_EC50" "CHEMBL204_Ki" "CHEMBL2147_Ki" "CHEMBL214_Ki" "CHEMBL218_EC50" "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki" "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50" "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki" "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki" "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki" "CHEMBL4616_EC50" "CHEMBL4792_Ki")
selected_tasks=("CHEMBL244_Ki" "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki" "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki" "CHEMBL4616_EC50" "CHEMBL4792_Ki")

for ((i=0; i<times; i++)); do
  seed=$((2024+i*10))
  for split_method in "${split_methods[@]}"; do
    for task in "${selected_tasks[@]}"; do
      echo "Current seed: $seed, current split method: $split_method, current task: $task"
      result_file="./result/GEM_${task}_${split_method}_${seed}_all_result.csv"
      if test -f ${result_file}; then
        echo "Result existed!"
      else
        echo "Result not existed!"
        
        # 设置 batch_size 为 16 当任务为 CHEMBL237_Ki 和 seed 为 2054
        if [[ "$task" == "CHEMBL237_Ki" && $seed -eq 2054 ]] || [[ "$task" == "CHEMBL3979_EC50" && $seed -eq 2054 ]]; then
          batch_size=16
        else
          batch_size=32  # 默认值
        fi

        if [[ " ${classification_tasks[@]} " =~ " ${task} " ]]; then
          python finetune_class2.py --task_name ${task} --data_name ${task}_${split_method}_${seed} \
                                    --data_path ./data/raw_data --processed_data_path ./data/processed_data --group_data_path ./data/group \
                                    --compound_encoder_config model_configs/geognn_l8.json --model_config model_configs/down_mlp2.json \
                                    --init_model ./pretrain_models-chemrl_gem/class.pdparams --model_dir ./model --batch_size ${batch_size}
        elif [[ " ${regression_tasks[@]} " =~ " ${task} " ]]; then
          python finetune_regr2.py --task_name ${task} --data_name ${task}_${split_method}_${seed} \
                                  --data_path ./data/raw_data --processed_data_path ./data/processed_data --group_data_path ./data/group \
                                  --compound_encoder_config model_configs/geognn_l8.json --model_config model_configs/down_mlp2.json \
                                  --init_model ./pretrain_models-chemrl_gem/regr.pdparams --model_dir ./model --batch_size ${batch_size}
        fi
      fi
    done
  done
done