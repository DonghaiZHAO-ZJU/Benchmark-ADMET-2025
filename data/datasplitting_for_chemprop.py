import os
import pandas as pd

classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","HIV_large","Tox21_NR_ER"]
regression_tasks = ["Caco2","HalfLife","VDss",'PAMPA1','CycPept_Caco2']
select_tasks = ['CycPept_Caco2']
split_methods=["random","scaffold","Perimeter"]
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