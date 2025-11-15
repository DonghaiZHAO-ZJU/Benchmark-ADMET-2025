#!/bin/bash
#SBATCH -J practice
#SBATCH -o practice-%j.log
#SBATCH -e practice-%j.err
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -w gpu18  # If you want to specify a computing node, you can write its name here and remove the first #

python finetune_new.py --config base --model_path ../models/pretrained/base/base.pth --dataset Caco2 --data_path ../datasets/ --dataset_type regression --metric r2 rmse mae --split random_2024 --weight_decay 0 --dropout 0 --lr 3e-5 --seed 2024 --use_scaler --scaler RobustScaler