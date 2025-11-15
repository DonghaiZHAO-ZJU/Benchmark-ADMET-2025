from experiment.downstream_task import train_K_BERT
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split_method", type=str, choices=["random", "scaffold", "Perimeter", "Maximum_Dissimilarity","MoleculeACE"], help="Name of the data")
parser.add_argument("--seed", type=int, choices=[2024, 2034, 2044, 2054, 2064], help="Name of the data")
parser.add_argument("--scaler", type=str, choices=['StandardScaler', 'PowerTransformer', 'RobustScaler'], help="Name of scaler")
args = parser.parse_args()

# os.makedirs('./model', exist_ok=True)
# os.makedirs('./result', exist_ok=True)

all_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability", 
             "Caco2","HalfLife","VDss","HIV_large"]
classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability"]
regression_tasks = ["Caco2","HalfLife","VDss","PAMPA1",'CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50', 
                    'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 
                    'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki', 
                    'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 
                    'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 
                    'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 
                    'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 
                    'CHEMBL4616_EC50', 'CHEMBL4792_Ki']
selected_tasks = ['CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50', 
                    'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 
                    'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki', 
                    'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 
                    'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 
                    'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 
                    'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 
                    'CHEMBL4616_EC50', 'CHEMBL4792_Ki']
for task in selected_tasks:
    if task in classification_tasks:
        train_K_BERT(task_name=task, data_name=f'{task}_{args.split_method}_{args.seed}', 
                     split_method=args.split_method, scaler=args.scaler, classification=True, savecheckpoint=True)
    else:
        train_K_BERT(task_name=task, data_name=f'{task}_{args.split_method}_{args.seed}', 
                     split_method=args.split_method, scaler=args.scaler, classification=False, savecheckpoint=True)