import numpy as np


import pandas as pd
import torch
from torch_geometric.data import Dataset

import os
from tqdm import tqdm
import deepchem as dc
import rdkit
from rdkit import Chem


def seed_everything(seed):
    """Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LoadSolDataset(Dataset):
    def __init__(self, root, raw_filename, transform=None, pre_transform=None):
        """
        root: directory of where raw file is at. Split into two path, processed and raw.
        filename: name of the raw file
        won't be using transform or pre-transform in this project
        """
        self.raw_filename = raw_filename
        super(LoadSolDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.raw_filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f"molecule_{i}.pt" for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["SMILES"])
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()
            data.y = self._get_label(row["logS"])
            data.smiles = row["SMILES"]
            torch.save(data, os.path.join(self.processed_dir, f"molecule_{idx}.pt"))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float32)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"molecule_{idx}.pt"))


class LoadDataset(Dataset):
    def __init__(self, root, data_name=None, split=None, transform=None, pre_transform=None):
        """
        root: directory of where raw file is at. Split into two path, processed and raw.
        filename: name of the raw file
        won't be using transform or pre-transform in this project
        """
        self.data_name=data_name
        self.split=split
        self.mean=None
        self.std=None
        super(LoadDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw', self.data_name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed', self.data_name, self.split)

    @property
    def raw_file_names(self):
        return [f"{self.data_name}_{self.split}.csv"]

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0])
        return [f"molecule_{i}.pt" for i in range(len(self.data))]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
        label_name = [column for column in self.data.columns if column != 'smiles'][-1]
        print(label_name)
        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["smiles"])
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()
            data.y = self._get_label(row[label_name])
            data.smiles = row["smiles"]
            torch.save(data, os.path.join(self.processed_dir, f"molecule_{idx}.pt"))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float32)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"molecule_{idx}.pt"))
    
    # def get_scaler(self):
    #     self.data = pd.read_csv(self.raw_paths[0])
    #     mean = np.mean(self.data[[column for column in self.data.columns if column != 'smiles'][-1]])
    #     std = np.std(self.data[[column for column in self.data.columns if column != 'smiles'][-1]])
    #     return mean, std

    def get_scaler(self):
        self.data = pd.read_csv(self.raw_paths[0])
        mean = np.median(self.data[[column for column in self.data.columns if column != 'smiles'][-1]])
        std = np.percentile(self.data[[column for column in self.data.columns if column != 'smiles'][-1]], 75) - np.percentile(self.data[[column for column in self.data.columns if column != 'smiles'][-1]], 25)
        return mean, std
