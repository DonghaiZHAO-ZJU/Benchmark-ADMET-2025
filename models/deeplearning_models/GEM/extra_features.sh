#!/bin/bash

split_methods=("random" "scaffold" "Perimeter" "Maximum_Dissimilarity")
all_tasks=("CHEMBL1862_Ki" "CHEMBL1871_Ki" "CHEMBL2034_Ki" "CHEMBL2047_EC50" "CHEMBL204_Ki" "CHEMBL2147_Ki" "CHEMBL214_Ki" "CHEMBL218_EC50" "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki" "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50" "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki" "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki" "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki" "CHEMBL4616_EC50" "CHEMBL4792_Ki")
classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "HIV_large")
regression_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss" "PAMPA1")


for task in "${all_tasks[@]}"; do

  python extract_features.py --task_name ${task} \
                            --data_path ./data/raw_data --processed_data_path ./data/processed_data  --group_data_path ./data/group \
                            --compound_encoder_config model_configs/geognn_l8.json  --model_config model_configs/down_mlp2.json  --init_model ./pretrain_models-chemrl_gem/class.pdparams  \
                            --model_dir ./model

done
