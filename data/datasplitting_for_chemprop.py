import os
import pandas as pd

classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","Tox21_NR_ER","CYP2C9_Substrate","CYP2D6_Inhibition","LinPept_CellPen","LinPept_NonFouling"]
regression_tasks = ["Caco2","HalfLife","VDss",'PAMPA1'
                    # Following are MoleculeACE tasks
                    'CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50', 
                    'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 
                    'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki', 
                    'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 
                    'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 
                    'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 
                    'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 
                    'CHEMBL4616_EC50', 'CHEMBL4792_Ki']
select_tasks = []
split_methods=["random","scaffold","Perimeter"] # Change to "MoleculeACE" if you want to use MoleculeACE tasks
for i in range(5):
    seed = 2024+i*10
    for split_policy in split_methods:
        for task in select_tasks:
            data_origin = pd.read_csv(f'data_with_group_{split_policy}/{task}_{split_policy}_{seed}.csv'.format(task))
            train_data = data_origin[data_origin['group']=='training'][['smiles', task]]
            valid_data = data_origin[data_origin['group']=='valid'][['smiles', task]]
            test_data = data_origin[data_origin['group']=='test'][['smiles', task]]
            print(len(train_data), len(valid_data), len(test_data))
            os.makedirs(f'data_split_for_chemprop/{task}_{split_policy}_{seed}/', exist_ok=True)
            train_data.to_csv(f'data_split_for_chemprop/{task}_{split_policy}_{seed}/{task}_{split_policy}_{seed}_training.csv', index=False)
            valid_data.to_csv(f'data_split_for_chemprop/{task}_{split_policy}_{seed}/{task}_{split_policy}_{seed}_valid.csv', index=False)
            test_data.to_csv(f'data_split_for_chemprop/{task}_{split_policy}_{seed}/{task}_{split_policy}_{seed}_test.csv', index=False)