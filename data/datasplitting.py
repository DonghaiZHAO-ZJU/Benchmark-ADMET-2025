import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit.Chem import AllChem
from utils.scaffold import scaffold_split

from mood.splitter import PerimeterSplit, MaxDissimilaritySplit, MOODSplitter

split_policys = ['random', 'scaffold', "Perimeter"] # change
classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","Tox21_NR_ER"]
regression_tasks = ["Caco2","HalfLife","VDss",'PAMPA1']
select_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","Tox21_NR_ER","Caco2","HalfLife","VDss",'PAMPA1']

for i in range(5):
    seed = 2024+i*10
    print(f'------ seed: {seed} ------', flush=True)
    for split_policy in split_policys:
        print(f'****** split policy: {split_policy} ******', flush=True)
        merged_data = pd.DataFrame()
        for task in select_tasks:
            print(f'###### task: {task} ######', flush=True)
            data_origin = pd.read_csv('data_after_processing/{}.csv'.format(task))
            if split_policy == 'random':
                if task in ["ClinTox","Tox21_NR_ER"]:
                    train_data, remain_data = train_test_split(data_origin, test_size=0.2, random_state=seed, stratify=data_origin[task])
                    valid_data, test_data = train_test_split(remain_data, test_size=0.5, random_state=seed, stratify=remain_data[task])
                elif task in ['constrained_logP']:
                    train_data, remain_data = train_test_split(data_origin, test_size=1/6, random_state=seed)
                    valid_data, test_data = train_test_split(remain_data, test_size=0.5, random_state=seed)   
                else:
                    train_data, remain_data = train_test_split(data_origin, test_size=0.2, random_state=seed)
                    valid_data, test_data = train_test_split(remain_data, test_size=0.5, random_state=seed)
                print(len(train_data), len(valid_data), len(test_data), flush=True)
            elif split_policy == 'scaffold':
                if task in ['constrained_logP']:
                    train_data, valid_data, test_data = scaffold_split(data_path=None, data=data_origin, size=[5/6, 1/12, 1/12], seed=seed)
                else:
                    train_data, valid_data, test_data = scaffold_split(data_path=None, data=data_origin, size=[0.8, 0.1, 0.1], seed=seed)
                print(len(train_data), len(valid_data), len(test_data), flush=True)
            elif split_policy == 'Perimeter':
                smiles = np.stack([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), nBits=2048, radius=2) for smiles in data_origin['smiles'].tolist()])
                candidate_splitters = {"Perimeter": PerimeterSplit(n_clusters=50, train_size=0.9, test_size=0.1, n_splits=1, random_state=seed)}
                mood_splitter = MOODSplitter(candidate_splitters, metric="jaccard")
                mood_splitter.fit(smiles, data_origin[task])
                for training, test in mood_splitter.split(smiles, data_origin[task]):
                    training_data = data_origin.loc[training].reset_index(drop=True)
                    test_data = data_origin.loc[test].reset_index(drop=True)
                smiles = np.stack([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), nBits=2048, radius=2) for smiles in training_data['smiles'].tolist()])
                candidate_splitters = {"Perimeter": PerimeterSplit(n_clusters=50, train_size=8/9, test_size=1/9, n_splits=1, random_state=seed)}
                mood_splitter = MOODSplitter(candidate_splitters, metric="jaccard")
                mood_splitter.fit(smiles, training_data[task])
                for train, valid in mood_splitter.split(smiles, training_data[task]):
                    train_data = training_data.loc[train].reset_index(drop=True)
                    valid_data = training_data.loc[valid].reset_index(drop=True)
                print(len(train_data), len(valid_data), len(test_data), flush=True)
            else:
                smiles = np.stack([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), nBits=2048, radius=2) for smiles in data_origin['smiles'].tolist()])
                candidate_splitters = {"Maximum Dissimilarity": MaxDissimilaritySplit(n_clusters=50, train_size=0.9, test_size=0.1, n_splits=1, random_state=seed)}
                mood_splitter = MOODSplitter(candidate_splitters, metric="jaccard")
                mood_splitter.fit(smiles, data_origin[task])
                for training, test in mood_splitter.split(smiles, data_origin[task]):
                    training_data = data_origin.loc[training].reset_index(drop=True)
                    test_data = data_origin.loc[test].reset_index(drop=True)
                smiles = np.stack([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), nBits=2048, radius=2) for smiles in training_data['smiles'].tolist()])
                candidate_splitters = {"Maximum Dissimilarity": MaxDissimilaritySplit(n_clusters=50, train_size=8/9, test_size=1/9, n_splits=1, random_state=seed)}
                mood_splitter = MOODSplitter(candidate_splitters, metric="jaccard")
                mood_splitter.fit(smiles, training_data[task])
                for train, valid in mood_splitter.split(smiles, training_data[task]):
                    train_data = training_data.loc[train].reset_index(drop=True)
                    valid_data = training_data.loc[valid].reset_index(drop=True)
                print(len(train_data), len(valid_data), len(test_data), flush=True)      
            train_data['group'] = 'training'
            valid_data['group'] = 'valid'
            test_data['group'] = 'test'
            result_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)
            os.makedirs('data_with_group_{}'.format(split_policy), exist_ok=True)
            result_data.to_csv('data_with_group_{}/{}_{}_{}.csv'.format(split_policy, task, split_policy, seed), index=False)