from dgl import DGLGraph
import pandas as pd
from rdkit.Chem import MolFromSmiles
import numpy as np
from dgl.data.graph_serialize import save_graphs, load_graphs, load_labels
import torch
from rdkit import Chem
import random
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os


def find_var(df, min_value):
    df1 = df.copy()
    df1.loc["var"] = np.var(df1.values, axis=0)
    col = [x for i, x in enumerate(df1.columns) if df1.iat[-1, i] < min_value]
    return col


def find_sum_0(df):
    '''
    input: df
    return: the columns of labels with no positive labels
    '''
    df1 = df.copy()
    df1.loc["sum"] = np.sum(df1.values, axis=0)
    col = [x for i, x in enumerate(df1.columns) if df1.iat[-1, i] == 0]
    return col


def find_sum_1(df):
    '''
    input: df
    return: the columns of labels with no negative labels
    '''
    df1 = df.copy()
    df1.loc["sum"] = np.sum(df1.values, axis=0)
    col = [x for i, x in enumerate(df1.columns) if df1.iat[-1, i] == len(df)]
    return col


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


def atom_features(atom, explicit_H=False, use_chirality=True):
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


def bond_features(bond, use_chirality=True, atompair=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    if atompair:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats = bond_feats + one_of_k_atompair_encoding(
            atom_pair_str, [['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                            ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                            ['CBr', 'BrC'], ['others']]
        )
    return np.array(bond_feats).astype(float)


def etype_features(bond, use_chirality=True):
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
    return index


def construct_attentivefp_bigraph_from_smiles(smiles):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    mol = MolFromSmiles(smiles)

    if mol is not None:
        try:
            num_atoms = mol.GetNumAtoms()
            g.add_nodes(num_atoms)
            atoms_feature_all = []
            for atom in mol.GetAtoms():
                atom_feature = atom_features(atom)
                atoms_feature_all.append(atom_feature)
            g.ndata["node"] = torch.tensor(atoms_feature_all)

            # Add edges
            src_list = []
            dst_list = []
            etype_feature_all = []
            num_bonds = mol.GetNumBonds()
            for i in range(num_bonds):
                bond = mol.GetBondWithIdx(i)
                bond_feature = bond_features(bond)
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                src_list.extend([u, v])
                dst_list.extend([v, u])
                etype_feature_all.append(bond_feature)
                etype_feature_all.append(bond_feature)

            g.add_edges(src_list, dst_list)

            g.edata["bond"] = torch.tensor(etype_feature_all)
            return g
        except:
            return 1
    else:
        return 1

def build_mol_graph_data(dataset_smiles, labels_name, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        g_attentivefp = construct_attentivefp_bigraph_from_smiles(smiles)
        if g_attentivefp != 1:
            molecule = [smiles, g_attentivefp, labels.loc[i], split_index.loc[i]]
            dataset_gnn.append(molecule)
            print('{}/{} molecule is transformed to mol graph! {} is transformed failed!'.format(i + 1, molecule_number,
                                                                                    len(failed_molecule)), flush=True)
        else:
            print('{} is transformed to mol graph failed!'.format(smiles), flush=True)
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)
    print('{}({}) is transformed to mol graph failed!'.format(failed_molecule, len(failed_molecule)), flush=True)
    return dataset_gnn

def built_mol_graph_data_and_save(
        origin_data_path='practice.csv',
        labels_name='label',
        save_g_path='No_name_mol_graph_dataset.csv',
        save_g_group_path='No_name_mol_graph_dataset_group.csv',
):
    data_origin = pd.read_csv(origin_data_path, index_col=None)
    smiles_name = 'smiles'
    data_set_gnn = build_mol_graph_data(dataset_smiles=data_origin, labels_name=labels_name,
                                        smiles_name=smiles_name)

    smiles,  g_attentivefp, labels, split_index = map(list, zip(*data_set_gnn))
    graph_labels = {'labels': torch.tensor(labels)
                    }
    split_index_pd = pd.DataFrame(columns=['smiles', 'group'])
    split_index_pd.smiles = smiles
    split_index_pd.group = split_index
    split_index_pd.to_csv(save_g_group_path, index=False, columns=None)
    print('Molecules graph is saved!', flush=True)
    save_graphs(save_g_path, g_attentivefp, graph_labels)

def load_data_for_attentivefp(
        g_path='g_attentivefp.bin',
        g_group_path='g_group_path.csv',
        task_type='classification',
        use_scaler=False,
        scaler=None
        ):
    data = pd.read_csv(g_group_path, index_col=None)
    smiles = data.smiles.values
    group = data.group.to_list()

    g_attentivefp, detailed_information = load_graphs(g_path)
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
        molecule = [smiles[i], g_attentivefp[i], [labels[i]], [1]]
        train_set.append(molecule)

    for i in val_index:
        molecule = [smiles[i], g_attentivefp[i], [labels[i]], [1]]
        val_set.append(molecule)

    for i in test_index:
        molecule = [smiles[i], g_attentivefp[i], [labels[i]], [1]]
        test_set.append(molecule)

    if task_type=='regression' and use_scaler:
        labels = list(zip(*train_set))[-2]
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


def build_mask(labels_list, mask_value=100):
    mask = []
    for i in labels_list:
        if i == mask_value:
            mask.append(0)
        else:
            mask.append(1)
    return mask


def multi_task_build_dataset(dataset_smiles, labels_list, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_list]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        g_attentivefp = construct_attentivefp_bigraph_from_smiles(smiles)
        if g_attentivefp != 1:
            mask = build_mask(labels.loc[i], mask_value=123456)
            molecule = [smiles, g_attentivefp, labels.loc[i], mask, split_index.loc[i]]
            dataset_gnn.append(molecule)
            print('{}/{} molecule is transformed! {} is transformed failed!'.format(i + 1, molecule_number,
                                                                                    len(failed_molecule)))
        else:
            print('{} is transformed failed!'.format(smiles))
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)
    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def built_data_and_save_for_splited(
        origin_path='G:/加密/Dataset/AttentionFP/ClinTox.csv',
        save_g_attentivefp_path='G:/加密/Dataset/AttentionFP/ClinTox.csv',
        group_path='G:/加密/Dataset/AttentionFP/ClinTox.csv',
        task_list_selected=None,
):
    '''
        origin_path: str
            origin csv data set path, including molecule name, smiles, task
        des_path: str
            csv data set containing a molecular descriptors, including molecule name, smiles, task, descriptors
        smiles_name: str
            smiles columns name
        notused_name: list
            a list of column names (except for labels, smiles, descriptors)
        is_descriptor: bool
            wether use descriptor
        save_path: str
            graph out put path
        smiles_path: str
            smiles out put path
        des_name_path: str
            descriptors name out put path
        task_list_selected: list
            a list of selected task
        descriptor_selected: list
            a list of selected descriptors
        '''
    data_origin = pd.read_csv(origin_path)
    smiles_name = 'smiles'
    data_origin = data_origin.fillna(123456)
    labels_list = [x for x in data_origin.columns if x not in ['smiles', 'group']]
    if task_list_selected is not None:
        labels_list = task_list_selected
    data_set_gnn = multi_task_build_dataset(dataset_smiles=data_origin, labels_list=labels_list,
                                            smiles_name=smiles_name)

    smiles, g_attentivefp, labels, mask, split_index = map(list, zip(*data_set_gnn))
    graph_labels = {'labels': torch.tensor(labels),
                    'mask': torch.tensor(mask)
                    }
    split_index_pd = pd.DataFrame(columns=['smiles', 'group'])
    split_index_pd.smiles = smiles
    split_index_pd.group = split_index
    split_index_pd.to_csv(group_path, index=False, columns=None)
    print('Molecules graph is saved!')
    save_graphs(save_g_attentivefp_path, g_attentivefp, graph_labels)


def standardization_np(data, mean, std):
    return (data - mean) / (std + 1e-10)


def re_standar_np(data, mean, std):
    return data * (std + 1e-10) + mean

def split_dataset_according_index(dataset, train_index, val_index, test_index, data_type='np'):
    if data_type =='pd':
        return pd.DataFrame(dataset[train_index]), pd.DataFrame(dataset[val_index]), pd.DataFrame(dataset[test_index])
    if data_type =='np':
        return dataset[train_index], dataset[val_index], dataset[test_index]

def load_graph_from_csv_bin_for_splited(
        bin_g_attentivefp_path='example.bin',
        group_path='example.csv',
        select_task_index=None):
    smiles = pd.read_csv(group_path, index_col=None).smiles.values
    group = pd.read_csv(group_path, index_col=None).group.to_list()
    graphs, detailed_information = load_graphs(bin_g_attentivefp_path)
    labels = detailed_information['labels']
    mask = detailed_information['mask']
    if select_task_index is not None:
        labels = labels[:, select_task_index]
        mask = mask[:, select_task_index]
    # calculate not_use index
    notuse_mask = torch.mean(mask.float(), 1).numpy().tolist()
    not_use_index = []
    for index, notuse in enumerate(notuse_mask):
        if notuse==0:
            not_use_index.append(index)
    train_index=[]
    val_index = []
    test_index = []
    for index, group_index in enumerate(group):
        if group_index=='training' and index not in not_use_index:
            train_index.append(index)
        if group_index=='valid' and index not in not_use_index:
            val_index.append(index)
        if group_index == 'test' and index not in not_use_index:
            test_index.append(index)
    graph_List = []
    for g in graphs:
        graph_List.append(g)
    graphs_np = np.array(graphs)
    train_smiles, val_smiles, test_smiles = split_dataset_according_index(smiles, train_index, val_index, test_index)
    train_labels, val_labels, test_labels = split_dataset_according_index(labels.numpy(), train_index, val_index,
                                                                          test_index, data_type='pd')
    train_mask, val_mask, test_mask = split_dataset_according_index(mask.numpy(), train_index, val_index, test_index,
                                                                    data_type='pd')
    train_graph, val_graph, test_graph = split_dataset_according_index(graphs_np, train_index, val_index, test_index)

    # delete the 0_pos_label and 0_neg_label
    task_number = train_labels.values.shape[1]

    train_set = []
    val_set = []
    test_set = []
    for i in range(len(train_index)):
        molecule = [train_smiles[i], train_graph[i], train_labels.values[i], train_mask.values[i]]
        train_set.append(molecule)

    for i in range(len(val_index)):
        molecule = [val_smiles[i], val_graph[i], val_labels.values[i], val_mask.values[i]]
        val_set.append(molecule)

    for i in range(len(test_index)):
        molecule = [test_smiles[i], test_graph[i], test_labels.values[i], test_mask.values[i]]
        test_set.append(molecule)
    print(len(train_set), len(val_set), len(test_set), task_number)
    return train_set, val_set, test_set, task_number