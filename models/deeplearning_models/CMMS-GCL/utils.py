
import os
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import numpy as np

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/data', dataset=None, data_name=None, 
                 xd=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):
        self.dataset = dataset
        self.data_name = data_name
        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.data_name)

    @property
    def processed_file_names(self):
        return [f'{self.data_name}_{self.dataset}.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of matabolic pathway types prediction
    # Inputs:
    # xd - list of SMILES
    # Y: list of labels
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, y,smile_graph):
        assert (len(xd) == len(y)), "The two lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index,smi_em= smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                smi_em=torch.Tensor(smi_em),
                                y=torch.FloatTensor([labels]))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def label_weight(self):
        tensor = self.data.y
        num_pos = (tensor == 1).sum().item()
        num_neg = (tensor == 0).sum().item()
        return torch.tensor([num_neg/(num_pos+0.00000001)])
    
    def calc_scaler(self):
        labels = self.data.y.numpy()
        mean = np.mean(labels)
        std = np.std(labels)
        return mean, std

    # def calc_scaler(self):
    #     labels = self.data.y.numpy()
    #     mean = np.median(labels)
    #     std = np.percentile(labels, 75) - np.percentile(labels, 25)
    #     return mean, std

def standardization_np(data, mean, std):
    return (data - mean) / (std + 1e-10)

def re_standar_np(data, mean, std):
    return data * (std + 1e-10) + mean

