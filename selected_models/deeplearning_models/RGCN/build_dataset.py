from dgl import DGLGraph
import pandas as pd
from rdkit.Chem import MolFromSmiles
import numpy as np
from dgl.data.graph_serialize import save_graphs, load_graphs
import torch
import os
from rdkit import Chem
import random

import time

# build mol graph dataset for rgcn model
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, use_chirality=True, explicit_H=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
        ]) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def one_of_k_atompair_encoding(x, allowable_set):
    for atompair in allowable_set:
        if x in atompair:
            x = atompair
            break
        else:
            if atompair == allowable_set[-1]:
                x = allowable_set[-1]
            else:
                continue
    return [x == s for s in allowable_set]


def etype_features(bond, use_chirality=True, atompair=True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, m in enumerate(bond_feats_1):
        if m == True:
            a = i

    bond_feats_2 = bond.GetIsConjugated()
    if bond_feats_2 == True:
        b = 1
    else:
        b = 0

    bond_feats_3 = bond.IsInRing
    if bond_feats_3 == True:
        c = 1
    else:
        c = 0

    index = a * 1 + b * 4 + c * 8
    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        for i, m in enumerate(bond_feats_4):
            if m == True:
                d = i
        index = index + d * 16
    if atompair == True:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats_5 = one_of_k_atompair_encoding(
            atom_pair_str, [['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                            ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                            ['CBr', 'BrC'], ['others']]
        )
        for i, m in enumerate(bond_feats_5):
            if m == True:
                e = i
        index = index + e*64
    return index


def construct_RGCN_mol_graph_from_smiles(smiles, use_chirality=True):
    g = DGLGraph()

    # Add nodes
    mol = MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    atoms_feature_all = []
    for atom in mol.GetAtoms():
        atom_feature = atom_features(atom, use_chirality)
        atoms_feature_all.append(atom_feature)
    g.ndata["node"] = torch.tensor(np.array(atoms_feature_all)) # add np.array()


    # Add edges
    src_list = []
    dst_list = []
    etype_feature_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = etype_features(bond, use_chirality)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.append(etype_feature)
        etype_feature_all.append(etype_feature)

    g.add_edges(src_list, dst_list)
    normal_all = []
    for i in etype_feature_all:
        normal = etype_feature_all.count(i)/len(etype_feature_all)
        normal = round(normal, 1)
        normal_all.append(normal)

    g.edata["edge"] = torch.tensor(etype_feature_all)
    g.edata["normal"] = torch.tensor(normal_all)
    return g


def build_mol_graph_data(dataset_smiles, labels_name, smiles_name, use_chirality=True):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        g_rgcn = construct_RGCN_mol_graph_from_smiles(smiles, use_chirality)
        if g_rgcn != 1:
            molecule = [smiles, g_rgcn, labels.loc[i], split_index.loc[i]]
            dataset_gnn.append(molecule)
            print('{}/{} molecule is transformed to mol graph! {} is transformed failed!'.format(i + 1, molecule_number,
                                                                                    len(failed_molecule)))
        else:
            print('{} is transformed to mol graph failed!'.format(smiles))
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)
    print('{}({}) is transformed to mol graph failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def built_mol_graph_data_and_save(
        origin_data_path='practice.csv',
        labels_name='label',
        save_g_path='No_name_mol_graph_dataset.csv',
        save_g_group_path='No_name_mol_graph_dataset_group.csv',
        use_chirality=True
):
    data_origin = pd.read_csv(origin_data_path, index_col=None)
    smiles_name = 'smiles'
    data_set_gnn = build_mol_graph_data(dataset_smiles=data_origin, labels_name=labels_name,
                                        smiles_name=smiles_name, use_chirality=use_chirality)

    smiles,  g_rgcn, labels, split_index = map(list, zip(*data_set_gnn))
    graph_labels = {'labels': torch.tensor(labels)
                    }
    split_index_pd = pd.DataFrame(columns=['smiles', 'group'])
    split_index_pd.smiles = smiles
    split_index_pd.group = split_index
    split_index_pd.to_csv(save_g_group_path, index=False, columns=None)
    print('Molecules graph is saved!')
    save_graphs(save_g_path, g_rgcn, graph_labels)

def standardization_np(data, mean, std):
    return (data - mean) / (std + 1e-10)
def re_standar_np(data, mean, std):
    return data * (std + 1e-10) + mean

def load_data_for_rgcn(
        g_path='g_rgcn.bin',
        g_group_path='g_group_path.csv',
        task_type='classification',
        use_scaler=False,
        scaler=None):
    data = pd.read_csv(g_group_path, index_col=None)
    smiles = data.smiles.values
    group = data.group.to_list()

    g_rgcn, detailed_information = load_graphs(g_path)
    labels = detailed_information['labels']

    # calculate not_use index
    train_index = []
    val_index = []
    test_index = []
    for index, group_index in enumerate(group):
        if group_index == 'training':
            train_index.append(index)
        if group_index == 'valid':
            val_index.append(index)
        if group_index == 'test':
            test_index.append(index)

    train_set = []
    val_set = []
    test_set = []

    for i in train_index:
        molecule = [smiles[i], g_rgcn[i], labels[i]]
        train_set.append(molecule)

    for i in val_index:
        molecule = [smiles[i], g_rgcn[i], labels[i]]
        val_set.append(molecule)

    for i in test_index:
        molecule = [smiles[i], g_rgcn[i], labels[i]]
        test_set.append(molecule)
    
    if task_type=='regression' and use_scaler:
        labels = list(zip(*train_set))[-1]
        if scaler=='StandardScaler':
            mean = np.nanmean(labels)
            std = np.nanstd(labels)
        if scaler=='RobustScaler':
            mean = np.nanmedian(labels)
            std = np.nanpercentile(labels, 75) - np.nanpercentile(labels, 25)
    else:
        mean = None
        std = None
    print('Training set: {}, mean: {}, std: {}; Validation set: {}; Test set: {}'.format(len(train_set), mean, std, len(val_set), len(test_set)))
    return train_set, val_set, test_set, mean, std

if __name__=='__main__':
    task_list = ['BBBP_random_2024']
    start_time = time.time()
    for task in task_list:
        input_csv = './data/origin_data/' + task + '.csv'
        output_g_dir = './data/graph_data/' + task
        os.makedirs(output_g_dir, exist_ok=True)
        output_g_path = output_g_dir + '/' + task + '.bin'
        output_g_group_dir = './data/graph_data/' + task
        output_g_group_path = output_g_group_dir + '/' + task  + '_group.csv'
        built_mol_graph_data_and_save(
            origin_data_path=input_csv,
            labels_name='BBBP',
            save_g_path=output_g_path,
            save_g_group_path=output_g_group_path,
        )
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))