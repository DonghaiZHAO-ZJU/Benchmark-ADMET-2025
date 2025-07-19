import tqdm
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs


# Borrowed from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
def standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)

def find_duplicates(lst):
    duplicates = []
    for item in lst:
        if lst.count(item) > 1 and item not in duplicates:
            duplicates.append(item)
    return len(duplicates), duplicates

def calculate_rsd(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    rsd = std_dev / mean
    return abs(rsd)

def mean_with_min_decimal_places(column):
    decimal_places = min([len(str(x).split('.')[1]) for x in column])
    mean = np.round(column.mean(), decimals=decimal_places)
    return mean

classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","Tox21_NR_ER"]
regression_tasks = ["Caco2","HalfLife","VDss","PAMPA1"]
select_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","Tox21_NR_ER","PAMPA1"]
for task in select_tasks:
    print("---------------------{}---------------------".format(task))
    data_origin = pd.read_csv("data/{}.csv".format(task), low_memory=False)
    smiles_list = data_origin['smiles'].tolist()
    label_list = data_origin[task].tolist()
    Canonical_smiles_list = []
    valid_label_list = []
    total = len(smiles_list) 
    for smiles, label in tqdm.tqdm(zip(smiles_list, label_list), total=total):
        with BlockLogs():
            try:
                new_smiles = standardize(smiles)
                mol=Chem.MolFromSmiles(new_smiles)
                if mol is None:
                    print('new smiles cannot be transfered to mol!', [smiles, new_smiles])
                else:
                    Canonical_smiles_list.append(new_smiles)
                    valid_label_list.append(label)
            except:
                print(f'{smiles} can not be transformed to new smiles! So drop it')
    data_new = pd.DataFrame()
    data_new["smiles"] = Canonical_smiles_list
    data_new[task] = valid_label_list
    num_duplicates, duplicate_elements = find_duplicates(Canonical_smiles_list)
    print("{} has {} identical molecules, they are {}".format(task, num_duplicates, duplicate_elements))
    if num_duplicates:
        m1=0
        m2=0
        n=0
        for Canonical_smiles in duplicate_elements:
            print('smiles ready to be processed:', Canonical_smiles)
            duplicate_rows = data_new[data_new["smiles"] == Canonical_smiles]
            column = duplicate_rows[task]
            num_unique_values = column.nunique()
            are_all_equal = num_unique_values == 1
            if are_all_equal:
                for index in duplicate_rows.index[1:]:
                    print(f"Deleted molecule: {data_new.loc[index, 'smiles']}")
                data_new = data_new.drop(duplicate_rows.index[1:]).reset_index(drop=True)
                print("{} has the same label {} and only save one".format(Canonical_smiles, len(duplicate_rows)))
                m1+=1
            elif task in regression_tasks and calculate_rsd(column) < 0.05:
                for index in duplicate_rows.index:
                    print(f"Deleted molecule: {data_new.loc[index, 'smiles']}")
                data_new = data_new.drop(duplicate_rows.index).reset_index(drop=True)
                new_row = pd.DataFrame({"smiles": [Canonical_smiles], task: [mean_with_min_decimal_places(column)]})
                data_new = pd.concat([data_new, new_row], ignore_index=True)
                print("{} has different labels {}, but they are very similar (Relative Error={}) so take average as their label".format(Canonical_smiles, len(duplicate_rows), calculate_rsd(column)))
                m2+=1
            else:
                for index in duplicate_rows.index:
                    print(f"Deleted molecule: {data_new.loc[index, 'smiles']}")
                data_new = data_new.drop(duplicate_rows.index).reset_index(drop=True)
                print("{} has different labels {} and drop all of them".format(Canonical_smiles, len(duplicate_rows)))
                n+=1
        print("{} identical molecules has been dropped to one".format(m1))
        print("There are {} similar molecules".format(m2))
        print("There are {} strange molecules".format(n))
        data_new.to_csv("data_after_processing2/{}.csv".format(task), index=False)
    else:
        data_new.to_csv("data_after_processing2/{}.csv".format(task), index=False)