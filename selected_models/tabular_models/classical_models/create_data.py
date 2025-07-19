import os
import argparse
import tqdm
import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

from mordred import Calculator, descriptors
from data_utils.pubchemfp import GetPubChemFPs
import torch

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

class rdkit_desc:
    """Calculate Descriptors using mordred"""
    def __init__(self, data_name):
        self.data_name = data_name
        dataframe = pd.read_csv(f'data/origin_data/{data_name}.csv')
        self.smiles = dataframe['smiles']
        self.labels_name = [column for column in dataframe.columns if column not in ['smiles','group']][0]
        self.labels = dataframe[self.labels_name]
        self.groups = dataframe['group']
        self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
        
    def compute_mordred(self, normalize=True, fill_missing=True):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header = calc.GetDescriptorNames()
        for i in tqdm.tqdm(range(len(self.mols))):
            ds = calc.CalcDescriptors(self.mols[i])
            rdkit_2d_desc.append(ds) 
        self.data_with_desc = pd.DataFrame(rdkit_2d_desc, columns=header, dtype=np.float64)       
        self.data_with_desc.insert(loc=0, column='smiles', value=self.smiles)
        self.data_with_desc.insert(loc=1, column=self.labels_name, value=self.labels)
        self.data_with_desc.insert(loc=2, column='group', value=self.groups)
        self.delete_col_with_all_zeros()
        self.delete_col_with_nan()
        if fill_missing:
            self.fill_missing_values()
        self.delete_col_with_zero_var()
        if normalize:
            self.standardize_features()
        os.makedirs(f'data/processed_data1/rdkit_desc/', exist_ok=True)
        self.data_with_desc.to_csv(f'data/processed_data1/rdkit_desc/{self.data_name}_rdkit_desc.csv', index=False)

    def delete_col_with_all_zeros(self):
        col_with_all_zeros = []
        for col in self.data_with_desc.columns:
            if col not in ['smiles', self.labels_name, 'group']:
                if (self.data_with_desc[col]==0).all():
                    col_with_all_zeros.append(col)
        print('col_with_all_zeros:',col_with_all_zeros)
        self.data_with_desc = self.data_with_desc.drop(col_with_all_zeros,axis=1)
    
    def delete_col_with_nan(self, missing_value_threshold=0.1):
        col_with_nan = []
        for col in self.data_with_desc.columns:
            if col not in ['smiles', self.labels_name, 'group']:
                missing_value_count = self.data_with_desc[col].isnull().sum()
                total_row_count = len(self.data_with_desc)
                missing_value_ratio = missing_value_count / total_row_count
                if missing_value_ratio > missing_value_threshold:
                    col_with_nan.append(col)
        print('col_with_nan:',col_with_nan)
        self.data_with_desc = self.data_with_desc.drop(col_with_nan,axis=1)

    def fill_missing_values(self, policy='KNN'):
        features = self.data_with_desc.columns.difference(['smiles', self.labels_name, 'group'])
        self.data_with_desc[features] = self.data_with_desc[features].clip(-1e9, 1e9)
        train_data = self.data_with_desc[self.data_with_desc['group'] == 'training']
        valid_data = self.data_with_desc[self.data_with_desc['group'] == 'valid']
        test_data = self.data_with_desc[self.data_with_desc['group'] == 'test']

        if policy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        else:
            imputer = KNNImputer(n_neighbors=5)

        # Fit imputer on training data and transform training, validation, and test data
        imputer.fit(train_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'training', features] = imputer.transform(train_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'valid', features] = imputer.transform(valid_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'test', features] = imputer.transform(test_data[features])
    
    def delete_col_with_zero_var(self):
        data = pd.DataFrame(self.data_with_desc.iloc[:, 3:].var(), columns=['variance']).T
        col_list_zero_var = []
        for col in data.columns:
            if (data[col] == 0).any():
                col_list_zero_var.append(col)
        print('col_list_zero_var:',col_list_zero_var)
        self.data_with_desc = self.data_with_desc.drop(col_list_zero_var,axis=1)

    def standardize_features(self):
        """Standardize the features using MinMax normalization"""
        scaler = MinMaxScaler()
        features = self.data_with_desc.columns.difference(['smiles', self.labels_name, 'group'])
        
        train_data = self.data_with_desc[self.data_with_desc['group'] == 'training']
        valid_data = self.data_with_desc[self.data_with_desc['group'] == 'valid']
        test_data = self.data_with_desc[self.data_with_desc['group'] == 'test']

        # Fit scaler on training data and transform training, validation, and test data
        scaler.fit(train_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'training', features] = scaler.transform(train_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'valid', features] = scaler.transform(valid_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'test', features] = scaler.transform(test_data[features])

class mordred_desc:
    """Calculate Descriptors using mordred"""
    def __init__(self, data_name):
        self.data_name = data_name
        dataframe = pd.read_csv(f'data/origin_data/{data_name}.csv')
        self.smiles = dataframe['smiles']
        self.labels_name = [column for column in dataframe.columns if column not in ['smiles','group']][0]
        self.labels = dataframe[self.labels_name]
        self.groups = dataframe['group']
        self.mols = [Chem.MolFromSmiles(i) for i in self.smiles]
        
    def compute_mordred(self, normalize=True, fill_missing=True):
        calc = Calculator(descriptors, ignore_3D=True)
        self.data_with_desc = calc.pandas(self.mols, nproc=4)
        self.data_with_desc = self.data_with_desc.astype(np.float64)
        self.data_with_desc.insert(loc=0, column='smiles', value=self.smiles)
        self.data_with_desc.insert(loc=1, column=self.labels_name, value=self.labels)
        self.data_with_desc.insert(loc=2, column='group', value=self.groups)
        self.delete_col_with_all_zeros()
        self.delete_col_with_nan()
        if fill_missing:
            self.fill_missing_values()
        self.delete_col_with_zero_var()
        if normalize:
            self.standardize_features()
        os.makedirs(f'data/processed_data1/mordred_desc/', exist_ok=True)
        self.data_with_desc.to_csv(f'data/processed_data1/mordred_desc/{self.data_name}_mordred_desc.csv', index=False)

    def delete_col_with_all_zeros(self):
        col_with_all_zeros = []
        for col in self.data_with_desc.columns:
            if col not in ['smiles', self.labels_name, 'group']:
                if (self.data_with_desc[col]==0).all():
                    col_with_all_zeros.append(col)
        print('col_with_all_zeros:',col_with_all_zeros)
        self.data_with_desc = self.data_with_desc.drop(col_with_all_zeros,axis=1)
    
    def delete_col_with_nan(self, missing_value_threshold=0.1):
        col_with_nan = []
        for col in self.data_with_desc.columns:
            if col not in ['smiles', self.labels_name, 'group']:
                missing_value_count = self.data_with_desc[col].isnull().sum()
                total_row_count = len(self.data_with_desc)
                missing_value_ratio = missing_value_count / total_row_count
                if missing_value_ratio > missing_value_threshold:
                    col_with_nan.append(col)
        print('col_with_nan:',col_with_nan)
        self.data_with_desc = self.data_with_desc.drop(col_with_nan,axis=1)

    def fill_missing_values(self, policy='KNN'):
        features = self.data_with_desc.columns.difference(['smiles', self.labels_name, 'group'])
        self.data_with_desc[features] = self.data_with_desc[features].clip(-1e9, 1e9)
        train_data = self.data_with_desc[self.data_with_desc['group'] == 'training']
        valid_data = self.data_with_desc[self.data_with_desc['group'] == 'valid']
        test_data = self.data_with_desc[self.data_with_desc['group'] == 'test']

        if policy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        else:
            imputer = KNNImputer(n_neighbors=5)

        # Fit imputer on training data and transform training, validation, and test data
        imputer.fit(train_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'training', features] = imputer.transform(train_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'valid', features] = imputer.transform(valid_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'test', features] = imputer.transform(test_data[features])
    
    def delete_col_with_zero_var(self):
        data = pd.DataFrame(self.data_with_desc.iloc[:, 3:].var(), columns=['variance']).T
        col_list_zero_var = []
        for col in data.columns:
            if (data[col] == 0).any():
                col_list_zero_var.append(col)
        print('col_list_zero_var:',col_list_zero_var)
        self.data_with_desc = self.data_with_desc.drop(col_list_zero_var,axis=1)

    def standardize_features(self):
        """Standardize the features using MinMax normalization"""
        scaler = MinMaxScaler()
        features = self.data_with_desc.columns.difference(['smiles', self.labels_name, 'group'])
        
        train_data = self.data_with_desc[self.data_with_desc['group'] == 'training']
        valid_data = self.data_with_desc[self.data_with_desc['group'] == 'valid']
        test_data = self.data_with_desc[self.data_with_desc['group'] == 'test']

        # Fit scaler on training data and transform training, validation, and test data
        scaler.fit(train_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'training', features] = scaler.transform(train_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'valid', features] = scaler.transform(valid_data[features])
        self.data_with_desc.loc[self.data_with_desc['group'] == 'test', features] = scaler.transform(test_data[features])

# data_builder = mordred_desc('Caco2_random_2024')
# data_builder.compute_mordred(normalize=True)

def get_rdkit_2d(smiles):
    generator = rdDescriptors.RDKit2D()
    features = generator.process(smiles)[1:]
    return features

def get_rdkit_2d_normalized(smiles):
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]
    return features

def get_maccskeys(mol): # 如何考虑手性?
    return np.array(MACCSkeys.GenMACCSKeys(mol))

def get_rdkfp(mol, minpath=1, maxpath=7, fpsize=2048): # 如何考虑手性?
    rdkfp = Chem.rdmolops.RDKFingerprint(mol, minPath=minpath, maxPath=maxpath, fpSize=fpsize)
    return np.array(rdkfp)

def get_morganfp(mol, rad=2, bits=2048, useChirality=True):
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad, nBits=bits, useChirality=useChirality)
    return np.array(morgan_fp)

def get_morgancount(mol, rad=2, bits=2048, useChirality=True):
    morgan_count_vec = AllChem.GetHashedMorganFingerprint(mol, radius=rad, nBits=bits, useChirality=useChirality)
    morgan_count = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(morgan_count_vec, morgan_count)
    return morgan_count

def get_phaERGfp(mol, minpath=1, maxpath=21, fuzzIncrement=0.3): # 如何考虑手性?
    fp_phaErGfp = AllChem.GetErGFingerprint(mol,minPath=minpath,maxPath=maxpath,fuzzIncrement=fuzzIncrement)
    return np.array(fp_phaErGfp)

def get_pubchemfp(mol): # 如何考虑手性?
    fp_pubcfp = GetPubChemFPs(mol)
    return fp_pubcfp

# def get_descriptors(data_name):
def get_fingerprint(data_name, fp_name):
    dataframe = pd.read_csv(f'data/origin_data/{data_name}.csv')
    label_name = [column for column in dataframe.columns if column not in ['smiles','group']][0]
    smiles_list = []
    labels = []
    groups = []
    fp_list = []
    for idx, row in tqdm.tqdm(dataframe.iterrows(), desc='calculating fingprint'):
        smiles = row['smiles']
        label = row[label_name]
        group = row['group']
        try:
            mol = Chem.MolFromSmiles(smiles)
            if fp_name=='MACCSkeys':
                fp = get_maccskeys(mol)
            elif fp_name =="RDKFP":
                fp = get_rdkfp(mol)
            elif fp_name=="MorganFP":
                fp = get_morganfp(mol)
            elif fp_name=="MorganCount":
                fp = get_morgancount(mol)
            elif fp_name=="MorganCount256":
                fp = get_morgancount(mol, rad=1, bits=256)
            elif fp_name=="MorganCount512":
                fp = get_morgancount(mol, bits=512)
            elif fp_name=="MorganCount1024":
                fp = get_morgancount(mol, bits=1024)      
            elif fp_name=="phaERGFP":
                fp = get_phaERGfp(mol)
            elif fp_name=="PubChemFP":
                fp = get_pubchemfp(mol)
        except Exception as e:
            print(f'{fp_name} error: ', str(e))
        smiles_list.append(smiles)
        labels.append(label)
        groups.append(group)
        fp_list.append(fp)
    data_with_fp = pd.DataFrame(fp_list)
    data_with_fp.insert(loc=0, column='smiles', value=smiles_list)
    data_with_fp.insert(loc=1, column=label_name, value=labels)
    data_with_fp.insert(loc=2, column='group', value=groups)
    os.makedirs(f'data/processed_data1/{fp_name}/', exist_ok=True)
    data_with_fp.to_csv(f'data/processed_data1/{fp_name}/{data_name}_{fp_name}.csv', index=False)

# get_fingerprint('Caco2_random_2024', 'rdkit_2d_desc_normalized')

class data_set(torch.utils.data.Dataset):
    def __init__(self, data_name, fp_name, data_type):
        super(data_set, self).__init__()
        try:
            data = pd.read_csv(f'data/processed_data/{fp_name}/{data_name}_{fp_name}.csv')
            self.data = data[data['group']==data_type].reset_index()
            self.fp_length = len(self.data.iloc[0, 4:].to_list())
        except Exception as e:
            print(e)
            get_fingerprint(data_name, fp_name)
            data = pd.read_csv(f'data/processed_data/{fp_name}/{data_name}_{fp_name}.csv')
            self.data = data[data['group']==data_type].reset_index()
            self.fp_length = len(self.data.iloc[0, 4:].to_list())

    
    def __getitem__(self, index):
        smiles = self.data.iloc[index, 1]
        label = self.data.iloc[index, 2]
        fingerprint = self.data.iloc[index, 4:].to_list()
        return [smiles, label, fingerprint]
        
    def __len__(self):
        return len(self.data)
    
    def get_label_weight(self):
        labels = self.data.iloc[:, 2].to_numpy()
        task_pos_weight_list = []
        num_pos = 0
        num_neg = 0
        for i in labels:
            if i == 1:
                num_pos = num_pos + 1
            if i == 0:
                num_neg = num_neg + 1
        weight = num_neg / (num_pos+0.00000001)
        task_pos_weight_list.append(weight)
        task_pos_weight = torch.tensor(task_pos_weight_list)
        return task_pos_weight 

    def data_standard(self, scaler_name):
        labels = self.data.iloc[:, 2].to_numpy()
        if scaler_name=='MinMaxScaler':
            Scaler = MinMaxScaler()
            Scaler.fit_transform(labels.reshape(-1, 1))
            print(f"data_min_: {Scaler.data_min_}")
            print(f"data_max_: {Scaler.data_max_}")
            print(f"scale_: {Scaler.scale_}")
            print(f"min_: {Scaler.min_}")
        if scaler_name=='StandardScaler':
            Scaler = StandardScaler()
            Scaler.fit_transform(labels.reshape(-1, 1))
            print(f"StandardScaler mean: {Scaler.mean_}")
            print(f"StandardScaler std: {Scaler.scale_}")
        if scaler_name=='RobustScaler':
            Scaler = RobustScaler()
            Scaler.fit_transform(labels.reshape(-1, 1))
            print(f"RobustScaler center: {Scaler.center_}")
            print(f"RobustScaler scale (IQR): {Scaler.scale_}")
        return Scaler              

def collate_fn(data):

    smiles_list, label_list, fp_list = map(list, zip(*data))

    return {
        'smiles':smiles_list,
        'label': torch.tensor(label_list).unsqueeze(dim=1),
        'fp':torch.tensor(fp_list)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--fp_name", type=str, required=True)
    args = parser.parse_args()
    if args.fp_name=='rdkit_desc':
        data_builder = rdkit_desc(args.data_name)
        data_builder.compute_mordred()
    elif args.fp_name=='mordred_desc':
        data_builder = mordred_desc(args.data_name)
        data_builder.compute_mordred()
    else:
        get_fingerprint(args.data_name, args.fp_name)
