#!/bin/bash

# 遇到错误继续执行
set +e

# 数据集分类
# classification_datasets=("BBBP" "hERG" "Mutagenicity" "oral_bioavailability" "HLM_metabolic_stability" "LinPept_NonFouling" "LinPept_CellPen")
classification_datasets=("LinPept_CellPen" "LinPept_NonFouling")
regression_datasets=("Caco2" "HalfLife" "VDss" "PAMPA1" "Macrocycle_PAMPA")
MoleculeACE_datasets=("CHEMBL1862_Ki" "CHEMBL1871_Ki" "CHEMBL2034_Ki" "CHEMBL2047_EC50" "CHEMBL204_Ki" "CHEMBL2147_Ki" "CHEMBL214_Ki" "CHEMBL218_EC50" "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki" "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50" "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki" "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki" "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki" "CHEMBL4616_EC50" "CHEMBL4792_Ki")

# 划分方式
split_methods=("random" "scaffold" "Perimeter")

# 随机种子
seeds=("2024" "2034" "2044" "2054" "2064")

# 结果目录
result_dir="./result"

# 检查任务是否已完成
is_completed() {
    local dataset=$1
    local split=$2
    local seed=$3
    local result_file="${result_dir}/KA-GNN_${dataset}_${split}_${seed}_all_result.csv"
    if [ -f "$result_file" ]; then
        return 0  # 已完成
    else
        return 1  # 未完成
    fi
}

# 检查并清理不完整的权重文件
clean_incomplete_weights() {
    local dataset=$1
    local split=$2
    local seed=$3
    local weight_prefix="./model_weights/${dataset}_${split}_${seed}"
    # 如果存在任意一个不完整的权重文件（不是5个完整的），则全部删除
    if [ -f "${weight_prefix}_1.pth" ] && ! [ -f "${weight_prefix}_2.pth" ] || \
       [ -f "${weight_prefix}_1.pth" ] && ! [ -f "${weight_prefix}_3.pth" ] || \
       [ -f "${weight_prefix}_1.pth" ] && ! [ -f "${weight_prefix}_4.pth" ] || \
       [ -f "${weight_prefix}_1.pth" ] && ! [ -f "${weight_prefix}_5.pth" ]; then
        rm -f "${weight_prefix}"_*.pth
        echo "Cleaned incomplete weights: $dataset $split $seed"
    fi
}

# 运行分类任务 (使用bce损失函数)
echo "Running classification tasks..."
for dataset in "${classification_datasets[@]}"; do
    for split in "${split_methods[@]}"; do
        for seed in "${seeds[@]}"; do
            # if is_completed "$dataset" "$split" "$seed"; then
            #     echo "Skip: $dataset $split $seed (already completed)"
            # else
                clean_incomplete_weights "$dataset" "$split" "$seed"
                echo "Running: python main.py --select_dataset $dataset --loss_sclect bce --split_method $split --seed $seed"
                python main.py --select_dataset $dataset --loss_sclect bce --split_method $split --seed $seed
            # fi
        done
    done
done

# 运行回归任务 (使用l2损失函数)
# echo "Running regression tasks..."
# for dataset in "${regression_datasets[@]}"; do
#     for split in "${split_methods[@]}"; do
#         for seed in "${seeds[@]}"; do
#             if is_completed "$dataset" "$split" "$seed"; then
#                 echo "Skip: $dataset $split $seed (already completed)"
#             else
#                 clean_incomplete_weights "$dataset" "$split" "$seed"
#                 echo "Running: python main.py --select_dataset $dataset --loss_sclect l2 --split_method $split --seed $seed"
#                 python main.py --select_dataset $dataset --loss_sclect l2 --split_method $split --seed $seed
#             fi
#         done
#     done
# done

# echo "Running moleculeace tasks..."
# for dataset in "${MoleculeACE_datasets[@]}"; do
#     for seed in "${seeds[@]}"; do
#         if is_completed "$dataset" "MoleculeACE" "$seed"; then
#             echo "Skip: $dataset MoleculeACE $seed (already completed)"
#         else
#             echo "Running: python main.py --select_dataset $dataset --loss_sclect l2 --split_method MoleculeACE --seed $seed"
#             python main.py --select_dataset $dataset --loss_sclect l2 --split_method MoleculeACE --seed $seed
#         fi
#     done
# done
echo "All tasks completed!"