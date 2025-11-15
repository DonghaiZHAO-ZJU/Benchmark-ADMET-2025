import os
import pickle
import pandas as pd
import scipy.sparse as sps
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import multiprocessing
import torch
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from molmcl.utils.data import *
from molmcl.utils.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

generator = RDKit2DNormalized()


def get_fingerprint_descriptor(smi):
    md = np.array(list(generator.process(smi)), dtype=float)[1:]
    md = np.where(np.isnan(md), 0, md)
    fp = np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(smi), minPath=1, maxPath=7, fpSize=512))
    return smi, md, fp


class MoleculeDataset(Dataset):
    def __init__(self, data_dir, data_name, feat_type, split_type, task):
        self.feat_type = feat_type
        self.split_type = split_type
        self.task = task

        # data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        # if 'CHEMBL' in data_name:
        #     smiles = data['smiles'].to_list()
        #     labels = data[['y']].values
        #
        # elif data_name == 'bbbp':
        #     smiles = data['smiles'].to_list()
        #     labels = data[['p_np']]
        #     labels = labels.replace(0, -1)
        #     labels = labels.values
        #
        # elif data_name == 'clintox':
        #     smiles = data['smiles'].to_list()
        #     labels = data[['FDA_APPROVED', 'CT_TOX']]
        #     labels = labels.replace(0, -1)
        #     labels = labels.values
        #
        # elif data_name == 'muv':
        #     smiles = data['smiles'].to_list()
        #     labels = data[['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
        #                    'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
        #                    'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']]
        #     labels = labels.replace(0, -1)
        #     labels = labels.fillna(0)
        #     labels = labels.values
        #
        # elif data_name == 'sider':
        #     smiles = data['smiles'].to_list()
        #     labels = data[['Hepatobiliary disorders',
        #                    'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
        #                    'Investigations', 'Musculoskeletal and connective tissue disorders',
        #                    'Gastrointestinal disorders', 'Social circumstances',
        #                    'Immune system disorders', 'Reproductive system and breast disorders',
        #                    'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        #                    'General disorders and administration site conditions',
        #                    'Endocrine disorders', 'Surgical and medical procedures',
        #                    'Vascular disorders', 'Blood and lymphatic system disorders',
        #                    'Skin and subcutaneous tissue disorders',
        #                    'Congenital, familial and genetic disorders',
        #                    'Infections and infestations',
        #                    'Respiratory, thoracic and mediastinal disorders',
        #                    'Psychiatric disorders', 'Renal and urinary disorders',
        #                    'Pregnancy, puerperium and perinatal conditions',
        #                    'Ear and labyrinth disorders', 'Cardiac disorders',
        #                    'Nervous system disorders',
        #                    'Injury, poisoning and procedural complications']]
        #     labels = labels.replace(0, -1)
        #     labels = labels.values
        #
        # elif data_name == 'toxcast':
        #     smiles = data['smiles'].to_list()
        #     labels = data[list(data.columns)[1:]]
        #     labels = labels.replace(0, -1)
        #     labels = labels.fillna(0)
        #     labels = labels.values
        #
        # elif data_name == 'tox21':
        #     smiles = data['smiles'].to_list()
        #     labels = data[['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        #                    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']]
        #     labels = labels.replace(0, -1)
        #     labels = labels.fillna(0)
        #     labels = labels.values
        #
        # elif data_name == 'bace':
        #     smiles = data['mol'].to_list()
        #     labels = data[['Class']]
        #     labels = labels.replace(0, -1)
        #     labels = labels.values
        #
        # elif data_name == 'hiv':
        #     smiles = data['smiles'].to_list()
        #     labels = data[['HIV_active']]
        #     labels = labels.replace(0, -1)
        #     labels = labels.values

        if 'CHEMBL' in data_name:
            data = pd.read_csv(os.path.join(data_dir, f'{data_name}_{split_type.split("_")[1]}.csv'))
        else:
            data = pd.read_csv(os.path.join(data_dir, f'{data_name}_{split_type}.csv'))

        smiles = data['smiles'].to_list()
        labels = data.iloc[:, [1]]

        if data_name in ['BBBP', 'HLM_metabolic_stability', 'Mutagenicity', 'oral_bioavailability', 'hERG']:
            assert self.task == 'classification'
            labels = labels.replace(0, -1)
            labels = labels.values
        else:
            assert self.task == 'regression'
            labels = labels.values

        splits = data['group'].to_list()

        # convert mol to graph with smiles validity filtering
        self.splits = []
        self.smiles, self.labels, self.mol_data = [], [], []
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                data = eval('mol_to_graph_data_obj_{}'.format(feat_type))(mol)
                self.smiles.append(smi)
                self.labels.append(labels[i])
                self.mol_data.append(self.transform(data))
                self.splits.append(splits[i])
        self.num_task = 1

        # get descriptors
        self.smi2md = {}
        self.smi2fp = {}
        pool = multiprocessing.Pool(16)
        for smi, md, fp in pool.imap_unordered(get_fingerprint_descriptor, self.smiles):
            self.smi2md[smi] = md
            self.smi2fp[smi] = fp

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        graph = self.mol_data[idx]
        graph.label = torch.Tensor(self.labels[idx])
        graph.smi = self.smiles[idx]
        graph.desc = torch.Tensor(self.smi2md[self.smiles[idx]])
        graph.fp = torch.Tensor(self.smi2fp[self.smiles[idx]])

        return graph
