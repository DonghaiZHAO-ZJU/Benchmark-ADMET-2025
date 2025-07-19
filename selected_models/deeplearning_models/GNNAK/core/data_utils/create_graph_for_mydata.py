import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from scipy.sparse import coo_matrix
import torch

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

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

def get_atom_features(atom, explicit_H = False, use_chirality=True):
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
        ]) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
             one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) + \
             one_of_k_encoding_unk(atom.GetHybridization(), [
             Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
             Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, 
             Chem.rdchem.HybridizationType.SP3D2,'other']) + [atom.GetIsAromatic()]

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

def bond_features(bond, use_stereochemistry=True, atompair=True):
    results = one_of_k_encoding(
        bond.GetBondType(), 
        [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]) + \
        [bond.GetIsConjugated(), bond.IsInRing()]
    edge_features_num = 6

    if use_stereochemistry:
        results += one_of_k_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        edge_features_num += 4

    if atompair:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        results += one_of_k_atompair_encoding(
            atom_pair_str, [['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                            ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                            ['CBr', 'BrC'], ['others']]
        )
        edge_features_num += 12
    return np.array(results), edge_features_num

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    num_atoms = mol.GetNumAtoms()

    atom_features = []
    for atom in mol.GetAtoms():
        feature = get_atom_features(atom)
        atom_features.append(feature / sum(feature))
    atom_features = np.array(atom_features)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edge_index.append([u, v])
        bond_feature = bond_features(mol.GetBondBetweenAtoms(u, v))[0]
        edge_attr.append(bond_feature / sum(bond_feature))
        edge_index.append([v, u])
        bond_feature = bond_features(mol.GetBondBetweenAtoms(v, u))[0]
        edge_attr.append(bond_feature / sum(bond_feature))
    edge_attr = np.array(edge_attr)
    return num_atoms, atom_features, edge_index, edge_attr