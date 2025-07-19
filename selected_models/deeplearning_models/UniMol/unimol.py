import argparse

parser = argparse.ArgumentParser(description='Uni-Mol')
parser.add_argument('--task', type=str, choices=["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","Caco2","HalfLife","VDss", "HIV_large", "PAMPA"], help='task')
parser.add_argument('--split_method', type=str, choices=["random", "scaffold", "Perimeter", "Maximum_Dissimilarity"], help='split_method')
parser.add_argument('--seed', type=int, help='seed')
parser.add_argument('--data_type', type=str,  choices=["classification", "regression"], help='data_type')
parser.add_argument('--metric', type=str, choices=["roc_auc", "r2"], help='metric')
args = parser.parse_args()

from unimol_tools import MolTrain

clf = MolTrain(task=args.data_type, 
                data_type='molecule', 
                epochs=100,
                learning_rate=1e-4,
                batch_size=64,
                early_stopping=20,
                metrics=args.metric,
                save_path=f'./model/{args.task}_{args.split_method}_{args.seed}',
                remove_hs=True,
                smiles_col='smiles',
                target_col_prefix=args.task,
                target_normalize='none',
                )           
result_pd = clf.fit(
                data_name = f'{args.task}_{args.split_method}_{args.seed}',
                data = f'./data/{args.task}_{args.split_method}_{args.seed}/{args.task}_{args.split_method}_{args.seed}_training.csv', 
                valid_data = f'./data/{args.task}_{args.split_method}_{args.seed}/{args.task}_{args.split_method}_{args.seed}_valid.csv',
                test_data = f'./data/{args.task}_{args.split_method}_{args.seed}/{args.task}_{args.split_method}_{args.seed}_test.csv')
result_pd.to_csv(f'./result/Uni-Mol_{args.task}_{args.split_method}_{args.seed}_all_result.csv', index=False)