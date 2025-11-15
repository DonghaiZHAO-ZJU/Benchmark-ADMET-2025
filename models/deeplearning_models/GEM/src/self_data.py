import os
from os.path import join, exists
import pandas as pd
from pahelix.datasets.inmemory_dataset import InMemoryDataset

def load_my_dataset(data_path, task_type):

    input_df = pd.read_csv(data_path)
    smiles_list = input_df['smiles']
    labels = input_df[[column for column in input_df.columns if column not in ['group', 'smiles']]]
    if task_type=='class':
        labels = labels.replace(0, -1)
    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset

class Splitter(object):
    """
    The abstract class of splitters which split up dataset into train/valid/test 
    subsets.
    """
    def __init__(self):
        super(Splitter, self).__init__()

class GroupSplitter(Splitter):
    def __init__(self):
        super(GroupSplitter, self).__init__()
    
    def split(self,
              dataset,
              group_dataframe):
        N = len(dataset)

        train_idx, valid_idx, test_idx = [], [], []
        for i in range(N):
            if group_dataframe[group_dataframe['smiles'] == dataset[i]['smiles']]['group'].item() == 'training':
                train_idx.append(i)
            elif group_dataframe[group_dataframe['smiles'] == dataset[i]['smiles']]['group'].item() == 'valid':
                valid_idx.append(i)
            else:
                test_idx.append(i)
        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset
