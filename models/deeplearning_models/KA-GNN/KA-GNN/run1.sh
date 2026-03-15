#!/bin/bash

# 数据集分类
classification_datasets=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability")
regression_datasets=("Caco2" "HalfLife" "VDss" "PAMPA1" "CycPept_Caco2")

# 划分方式
split_methods=("random" "scaffold" "Perimeter")

# 随机种子
seeds=("2024")

# 运行分类任务 (使用bce损失函数)
echo "Running classification tasks..."
for dataset in "${classification_datasets[@]}"; do
    for split in "${split_methods[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running: python main.py --select_dataset $dataset --loss_sclect bce --split_method $split --seed $seed"
            python main.py --select_dataset $dataset --loss_sclect bce --split_method $split --seed $seed
        done
    done
done

# 运行回归任务 (使用l2损失函数)
echo "Running regression tasks..."
for dataset in "${regression_datasets[@]}"; do
    for split in "${split_methods[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running: python main.py --select_dataset $dataset --loss_sclect l2 --split_method $split --seed $seed"
            python main.py --select_dataset $dataset --loss_sclect l2 --split_method $split --seed $seed
        done
    done
done

echo "All tasks completed!"