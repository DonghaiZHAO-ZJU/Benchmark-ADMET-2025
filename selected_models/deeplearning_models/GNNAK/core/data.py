import torch
import pickle, os, numpy as np
import scipy.io as sio
# from math import comb
from scipy.special import comb
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected
import networkx as nx

# two more simulation dataset from PNA and SMP paper
from core.data_utils.data_pna import GraphPropertyDataset
from core.data_utils.data_cycles import CyclesDataset
from core.data_utils.sbm_cliques import CliqueSBM
from core.data_utils.tudataset_gin_split import TUDatasetGINSplit

class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        a=sio.loadmat(self.raw_paths[0])
        self.train_idx = torch.from_numpy(a['train_idx'][0])
        self.val_idx = torch.from_numpy(a['val_idx'][0]) 
        self.test_idx = torch.from_numpy(a['test_idx'][0]) 

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            tri=np.trace(A3)/6
            tailed=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(np.trace(A3.dot(a))+np.trace(A2)-2*A2.sum())
            cus= a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg=a.sum(0)
            star=0
            for j in range(a.shape[0]):
                star+=comb(int(deg[j]),3)

            expy=torch.tensor([[tri,tailed,star,cyc4,cus]])

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1).long() # change to category
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, average # edges per graph: {ave_num_edges}.')

from core.data_utils.create_graph_for_mydata import smile_to_graph
import pandas as pd
class mydata(InMemoryDataset):
    def __init__(self, root='../data/admet/', data_name=None, split=None, transform=None, pre_transform=None):
        self.data_name=data_name
        self.split=split
        super(mydata, self).__init__(root, transform, pre_transform)
        path=os.path.join(self.processed_dir, f'{data_name}_{split}.pt')
        self.data, self.slices=torch.load(path)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed', self.data_name)

    @property
    def raw_file_names(self):
        return [f"{self.data_name}.csv"]

    @property
    def processed_file_names(self):
        return [f'{self.data_name}_training.pt', f'{self.data_name}_valid.pt', f'{self.data_name}_test.pt']

    def download(self):
        pass

    def process(self):
        data_origin = pd.read_csv(self.raw_paths[0])
        graphs={'training': [], 'valid': [], 'test': []}
        for _, row in data_origin.iterrows():
            smiles = row['smiles']
            c_size, features, edge_index, edge_attr = smile_to_graph(smiles)
            label = row[list(filter(lambda x: x not in ['smiles', 'group'], data_origin.columns))].values[0]
            group = row['group']
            graphdata = Data(x=torch.Tensor(features),
                             edge_index=torch.tensor(edge_index, dtype=torch.long).transpose(1, 0),
                             edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                             smi_em=None,
                             y=torch.Tensor([label]))
            graphdata.__setitem__('c_size', torch.LongTensor([c_size]))
            graphs[group].append(graphdata)
        
        for key, data_list in graphs.items():
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), os.path.join(self.processed_dir, f'{self.data_name}_{key}.pt'))
    
    def label_weight(self):
        tensor = self.data.y
        num_pos = (tensor == 1).sum().item()
        num_neg = (tensor == 0).sum().item()
        return num_neg/(num_pos+0.00000001)

    def get_scaler(self):
        mean = np.mean(self.data.y.numpy())
        std = np.std(self.data.y.numpy())
        return mean, std

    # def get_scaler(self):
    #     mean = np.median(self.data.y.numpy())
    #     std = np.percentile(self.data.y.numpy(), 75) - np.percentile(self.data.y.numpy(), 25)
    #     return mean, std


if __name__ == "__main__":
    # dataset = PlanarSATPairsDataset('data/EXP')
    dataset = GraphCountDataset('data/subgraphcount')
    print(dataset.data.x.max(), dataset.data.x.min())