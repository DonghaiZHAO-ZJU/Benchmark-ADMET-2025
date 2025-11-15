#!/bin/bash
#SBATCH -J practice
#SBATCH -o practice-%j.log
#SBATCH -e practice-%j.err
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
##SBATCH --gres=gpu:1
##SBATCH -w gpu19  # If you want to specify a computing node, you can write its name here and remove the first #

python build_dataset_for_tasks.py