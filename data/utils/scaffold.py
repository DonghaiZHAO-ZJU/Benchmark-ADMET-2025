import random
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(mol,include_chirality=False):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def scaffold_to_smiles(mol,use_indices=False):
    scaffolds = defaultdict(set)
    for i, one in enumerate(mol):
        scaffold = generate_scaffold(one)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(one)
    return scaffolds

def scaffold_split(data_path: str, data: pd.DataFrame, size, seed, return_index=False):
    assert sum(size) == 1

    if data_path is not None:
        df = pd.read_csv(data_path)
    if data is not None:
        df = data
    smiles_list = df['smiles'].tolist()
    # Split
    train_size, val_size, test_size = size[0] * len(df), size[1] * len(df), size[2] * len(df)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(smiles_list, use_indices=True)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
              f'train scaffolds = {train_scaffold_count:,} | '
              f'val scaffolds = {val_scaffold_count:,} | '
              f'test scaffolds = {test_scaffold_count:,}')
    
    # Map from indices to data
    train_set = df.loc[train].copy()
    val_set = df.loc[val].copy()
    test_set = df.loc[test].copy()
    if return_index==False:
        return train_set, val_set, test_set
    else:
        return train, val, test

if __name__ == "__main__":
    train_set, val_set, test_set = scaffold_split('data/BBBP.csv', [0.8, 0, 0.2], 2024)
    train_set.to_csv('scaffold_data/BBBP_train.csv', index=False)
    test_set.to_csv('scaffold_data/BBBP_test.csv', index=False)
    # train_set['group'] = 'train'
    # val_set['group'] = 'valid'
    # test_set['group'] = 'test'
    # logD = pd.concat([train_set, test_set])
    # os.makedirs('scaffold_data', exist_ok=True)
    # logD.to_csv('scaffold_data/logD_seed2023.csv', index=False)