import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.scaffold import scaffold_split

def get_origin_idx(train_smiles):
    train_idx = []
    for smiles in train_smiles:
        try:
            idx = list(data_origin['smiles']).index(smiles)
            train_idx.append(idx)
        except ValueError:
            pass
    return train_idx

classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","HIV_large", "ClinTox","Tox21_NR_ER"]
regression_tasks = ["Caco2","HalfLife","VDss",'PAMPA1','CycPept_Caco2','constrained_logP','constrained_logP_0.1',
                    'CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50', 
                    'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 
                    'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki', 
                    'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 
                    'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 
                    'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 
                    'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 
                    'CHEMBL4616_EC50', 'CHEMBL4792_Ki']
select_tasks = ['CycPept_Caco2']
split_methods=["random","scaffold","Perimeter"]

for i in range(5):
    seed = 2024+i*10
    for split_policy in split_methods:
        for task in select_tasks:
            data_origin = pd.read_csv('data_after_processing/{}.csv'.format(task))
            if split_policy == 'random':
                if task in ["ClinTox","Tox21_NR_ER"]:
                    train_data, remain_data = train_test_split(np.arange(len(data_origin)), test_size=0.2, random_state=seed, stratify=data_origin[task])
                    train_data1, remain_data1 = train_test_split(data_origin, test_size=0.2, random_state=seed, stratify=data_origin[task])
                    valid_data, test_data = train_test_split(remain_data, test_size=0.5, random_state=seed, stratify=remain_data1[task])
                elif task in ['constrained_logP']:
                    train_data, remain_data = train_test_split(np.arange(len(data_origin)), test_size=1/6, random_state=seed)
                    valid_data, test_data = train_test_split(remain_data, test_size=0.5, random_state=seed)   
                else:
                    train_data, remain_data = train_test_split(np.arange(len(data_origin)), test_size=0.2, random_state=seed)
                    valid_data, test_data = train_test_split(remain_data, test_size=0.5, random_state=seed)
                print(len(train_data), len(valid_data), len(test_data))
                train_idx_array = np.array(train_data, dtype=np.int64)
                valid_idx_array = np.array(valid_data, dtype=np.int64)
                test_idx_array = np.array(test_data, dtype=np.int64)
                data_list = np.asarray([train_idx_array, valid_idx_array, test_idx_array], dtype=object)
                os.makedirs(f'data_split_for_kpgt/{task}', exist_ok=True)
                np.save(f'data_split_for_kpgt/{task}/{split_policy}_{seed}.npy', data_list, allow_pickle=True)
            elif split_policy == 'scaffold':
                if task in ['constrained_logP']:
                    train_data, valid_data, test_data = scaffold_split(data_path=None, data=data_origin, size=[5/6, 1/12, 1/12], seed=seed, return_index=True)
                else:
                    train_data, valid_data, test_data = scaffold_split(data_path=None, data=data_origin, size=[0.8, 0.1, 0.1], seed=seed, return_index=True)
                train_idx_array = np.array(train_data, dtype=np.int64)
                valid_idx_array = np.array(valid_data, dtype=np.int64)
                test_idx_array = np.array(test_data, dtype=np.int64)
                data_list = np.asarray([train_idx_array, valid_idx_array, test_idx_array], dtype=object)
                np.save(f'data_split_for_kpgt/{task}/{split_policy}_{seed}.npy', data_list, allow_pickle=True)
                print(len(train_data), len(valid_data), len(test_data))
            elif split_policy == 'Perimeter':
                data_splitted = pd.read_csv(f'data_with_group_Perimeter/{task}_Perimeter_{seed}.csv')
                train_smiles = data_splitted[data_splitted['group'] == 'training']['smiles'].tolist()
                train_data = get_origin_idx(train_smiles)
                valid_smiles = data_splitted[data_splitted['group'] == 'valid']['smiles'].tolist()
                valid_data = get_origin_idx(valid_smiles)
                test_smiles = data_splitted[data_splitted['group'] == 'test']['smiles'].tolist()
                test_data = get_origin_idx(test_smiles)
                train_idx_array = np.array(train_data, dtype=np.int64)
                valid_idx_array = np.array(valid_data, dtype=np.int64)
                test_idx_array = np.array(test_data, dtype=np.int64)
                data_list = np.asarray([train_idx_array, valid_idx_array, test_idx_array], dtype=object)
                np.save(f'data_split_for_kpgt/{task}/{split_policy}_{seed}.npy', data_list, allow_pickle=True)
                print(len(train_data), len(valid_data), len(test_data))
            elif split_policy == 'Maximum_Dissimilarity':
                data_splitted = pd.read_csv(f'data_with_group_Maximum_Dissimilarity/{task}_Maximum_Dissimilarity_{seed}.csv')
                train_smiles = data_splitted[data_splitted['group'] == 'training']['smiles'].tolist()
                train_data = get_origin_idx(train_smiles)
                valid_smiles = data_splitted[data_splitted['group'] == 'valid']['smiles'].tolist()
                valid_data = get_origin_idx(valid_smiles)
                test_smiles = data_splitted[data_splitted['group'] == 'test']['smiles'].tolist()
                test_data = get_origin_idx(test_smiles)
                train_idx_array = np.array(train_data, dtype=np.int64)
                valid_idx_array = np.array(valid_data, dtype=np.int64)
                test_idx_array = np.array(test_data, dtype=np.int64)
                data_list = np.asarray([train_idx_array, valid_idx_array, test_idx_array], dtype=object)
                np.save(f'data_split_for_kpgt/{task}/{split_policy}_{seed}.npy', data_list, allow_pickle=True)
                print(len(train_data), len(valid_data), len(test_data))
            elif split_policy == 'MoleculeACE':
                data_splitted = pd.read_csv(f'data_with_group_MoleculeACE/{task}_MoleculeACE_{seed}.csv')
                train_smiles = data_splitted[data_splitted['group'] == 'training']['smiles'].tolist()
                train_data = get_origin_idx(train_smiles)
                valid_smiles = data_splitted[data_splitted['group'] == 'valid']['smiles'].tolist()
                valid_data = get_origin_idx(valid_smiles)
                test_smiles = data_splitted[data_splitted['group'] == 'test']['smiles'].tolist()
                test_data = get_origin_idx(test_smiles)
                train_idx_array = np.array(train_data, dtype=np.int64)
                valid_idx_array = np.array(valid_data, dtype=np.int64)
                test_idx_array = np.array(test_data, dtype=np.int64)
                data_list = np.asarray([train_idx_array, valid_idx_array, test_idx_array], dtype=object)
                os.makedirs(f"data_split_for_kpgt/MoleculeACE/{task}/splits/", exist_ok=True)
                np.save(f'data_split_for_kpgt/MoleculeACE/{task}/splits/{split_policy}_{seed}.npy', data_list, allow_pickle=True)
                data_origin.to_csv(f"data_split_for_kpgt/MoleculeACE/{task}/{task}.csv", index=False)
                print(len(train_data), len(valid_data), len(test_data))