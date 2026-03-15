#!/bin/bash

times=5
split_methods=("MoleculeACE")
all_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "Caco2" "HalfLife" "VDss" "PAMPA" 'constrained_logP')
classification_tasks=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability")
regression_tasks=("Caco2" "HalfLife" "VDss" "PAMPA1" 'constrained_logP' "CHEMBL1862_Ki" "CHEMBL1871_Ki" "CHEMBL2034_Ki" "CHEMBL2047_EC50" "CHEMBL204_Ki" "CHEMBL2147_Ki" "CHEMBL214_Ki" "CHEMBL218_EC50" "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki" "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50" "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki" "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki" "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki" "CHEMBL4616_EC50" "CHEMBL4792_Ki")
selected_tasks=("CHEMBL1862_Ki" "CHEMBL1871_Ki" "CHEMBL2034_Ki" "CHEMBL2047_EC50" "CHEMBL204_Ki" "CHEMBL2147_Ki" "CHEMBL214_Ki" "CHEMBL218_EC50" "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki" "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50" "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki" "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki" "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki" "CHEMBL4616_EC50" "CHEMBL4792_Ki")
for task in "${selected_tasks[@]}"; do
  for ((i=0; i<times; i++)); do
    seed=$((2024+i*10))
    for split_method in "${split_methods[@]}"; do
      echo "Current seed: $seed, current split method: $split_method, current task: $task"
      var_name="${split_method}_${seed}"
      result_file="../result/KPGT_${task}_${split_method}_${seed}_all_result.csv"
      if test -f ${result_file} ; then
        echo "Result existed!"
      else
        if [[ " ${classification_tasks[@]} " =~ " ${task} " ]]; then
          python finetune_new.py --config base --model_path ../models/pretrained/base/base.pth --dataset $task --data_path ../datasets/ --dataset_type classification --metric rocauc prauc acc --split "$var_name" --weight_decay 0 --dropout 0 --lr 3e-5 --seed 2024
        elif [[ " ${regression_tasks[@]} " =~ " ${task} " ]]; then
          python finetune_new.py --config base --model_path ../models/pretrained/base/base.pth --dataset $task --data_path ../datasets/ --dataset_type regression --metric r2 rmse mae --split "$var_name" --weight_decay 0 --dropout 0 --lr 3e-5 --seed 2024
        else
          echo "Task $task is not defined as classification or regression task."
        fi
      fi
    done
  done
done