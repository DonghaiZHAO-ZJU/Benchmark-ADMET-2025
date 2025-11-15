from experiment.downstream_task import train_K_BERT
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split_method", type=str, choices=["random", "scaffold", "Perimeter", "Maximum_Dissimilarity"], help="Name of the data")
parser.add_argument("--seed", type=int, choices=[2024, 2034, 2044, 2054, 2064], help="Name of the data")
parser.add_argument("--scaler", type=str, choices=['StandardScaler', 'PowerTransformer', 'RobustScaler'], help="Name of scaler")
args = parser.parse_args()

# os.makedirs('./model', exist_ok=True)
# os.makedirs('./result', exist_ok=True)

all_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability", 
             "Caco2","HalfLife","VDss","HIV_large"]
classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","HIV_large"]
regression_tasks = ["Caco2","HalfLife","VDss","PAMPA1"]
selected_tasks = ["PAMPA1"]
for task in selected_tasks:
    if task in classification_tasks:
        train_K_BERT(task_name=task, data_name=f'{task}_{args.split_method}_{args.seed}', 
                     split_method=args.split_method, scaler=args.scaler, classification=True, savecheckpoint=True)
    else:
        train_K_BERT(task_name=task, data_name=f'{task}_{args.split_method}_{args.seed}', 
                     split_method=args.split_method, scaler=args.scaler, classification=False, savecheckpoint=True)