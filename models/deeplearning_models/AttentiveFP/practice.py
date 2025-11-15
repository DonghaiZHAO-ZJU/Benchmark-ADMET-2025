from AttentiveFP_singletask_model import AttentiveFP_model
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split_method", type=str, choices=["random", "scaffold", "Perimeter", "Maximum_Dissimilarity"], help="Name of the data")
parser.add_argument("--seed", type=int, choices=[2024, 2034, 2044, 2054, 2064], help="Name of the data")
parser.add_argument("--scaler", type=str, choices=['StandardScaler', 'PowerTransformer', 'RobustScaler'], help="Name of the data")
args = parser.parse_args()

all_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","Caco2","HalfLife","VDss","HIV_large","PAMPA"]
classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability"]
regression_tasks = ["Caco2","HalfLife","VDss","PAMPA", "PAMPA1"]
selected_tasks = ["PAMPA1"]

for task in selected_tasks:
    if task in classification_tasks:
        AttentiveFP_model(times=5, task_name=task, data_name=f'{task}_{args.split_method}_{args.seed}', use_scaler=False, scaler=None, classification=True)
    else:
        AttentiveFP_model(times=5, task_name=task, data_name=f'{task}_{args.split_method}_{args.seed}', use_scaler=True, scaler=args.scaler, classification=False)