#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:08:37 2024

@author: longlee
"""

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import random
import dgl
import statistics
import csv
import time

from sklearn.preprocessing import StandardScaler, RobustScaler

from logzero import logger
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn.metrics import precision_recall_curve, auc

from sklearn import metrics
from model.ka_gnn import KA_GNN,KA_GNN_two
from model.mlp_sage import MLPGNN,MLPGNN_two
from model.kan_sage import KANGNN, KANGNN_two
from torch.optim.lr_scheduler import StepLR
from ruamel.yaml import YAML
from utils.splitters import ScaffoldSplitter
from utils.graph_path import path_complex_mol
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

from tqdm import tqdm


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list):
        self.labels = label_list
        self.graphs = graph_list
        self.device = torch.device('cpu') 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index].to(self.device)
        

        graph = self.graphs[index].to(self.device)
        
        return label, graph
    


def collate_fn(batch):
    labels, graphs = zip(*batch) 

    labels = torch.stack(labels)

    batched_graph = dgl.batch(graphs)

    return labels, batched_graph



def has_node_with_zero_in_degree(graph):
    if (graph.in_degrees() == 0).any():
                return True
    return False




def is_file_in_directory(directory, target_file):
    file_path = os.path.join(directory, target_file)
    return os.path.isfile(file_path)


#others
def get_label():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['label']


#tox21,12     
def get_tox():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

#clintox,2
def get_clintox():
    
    return ['FDA_APPROVED', 'CT_TOX']

#sider,27
def get_sider():

    return ['Hepatobiliary disorders',
           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
           'Investigations', 'Musculoskeletal and connective tissue disorders',
           'Gastrointestinal disorders', 'Social circumstances',
           'Immune system disorders', 'Reproductive system and breast disorders',
           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
           'General disorders and administration site conditions',
           'Endocrine disorders', 'Surgical and medical procedures',
           'Vascular disorders', 'Blood and lymphatic system disorders',
           'Skin and subcutaneous tissue disorders',
           'Congenital, familial and genetic disorders',
           'Infections and infestations',
           'Respiratory, thoracic and mediastinal disorders',
           'Psychiatric disorders', 'Renal and urinary disorders',
           'Pregnancy, puerperium and perinatal conditions',
           'Ear and labyrinth disorders', 'Cardiac disorders',
           'Nervous system disorders',
           'Injury, poisoning and procedural complications']

#muv
def get_muv():
    
    return ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692',
            'MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846',
            'MUV-852',	'MUV-858','MUV-859']




def creat_data(datafile, split_method, seed, encoder_atom, encoder_bond,batch_size,train_ratio,vali_ratio,test_ratio, loss_type='bce'):


    datasets = datafile

    directory_path = 'data/processed_data/'
    target_file_name = f"{datasets}_{split_method}_{seed}" +'.pth'

    if is_file_in_directory(directory_path, target_file_name):

        return True

    else:

        df = pd.read_csv('data/origin_data/' + f"{datasets}_{split_method}_{seed}" + '.csv')#

        if datasets in ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","Caco2","HalfLife","VDss",'PAMPA1',"LinPept_NonFouling","LinPept_CellPen",'CycPept_Caco2',
                        'CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50', 
                        'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 
                        'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki', 
                        'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 
                        'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 
                        'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 
                        'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 
                        'CHEMBL4616_EC50', 'CHEMBL4792_Ki', 
                        "Macrocycle_PAMPA"]:
            smiles_list, labels, groups = df['smiles'], df[datasets], df['group']
        else:
            # 对于其他数据集，如果没有groups列，使用传统方法
            smiles_list, labels = df['smiles'], df[get_label()]
            groups = None

        # 对回归任务进行标签归一化
        if loss_type != 'bce':  # 回归任务
            # 将标签转换为numpy数组
            labels_np = labels.values

            # 根据数据集选择合适的归一化方法
            if datasets == 'PAMPA1':
                # 对PAMPA1使用RobustScaler
                scaler = RobustScaler()
            else:
                # 对其他回归任务使用StandardScaler
                scaler = StandardScaler()

            # 只对训练集进行拟合
            # 首先需要按照groups划分数据来确定训练集
            train_labels_idx = []
            if groups is not None:
                for i, group in enumerate(groups):
                    if group == 'training':
                        train_labels_idx.append(i)
            else:
                # 如果没有groups，使用训练集比例
                train_size = int(len(labels) * train_ratio)
                train_labels_idx = list(range(train_size))

            # 对训练标签进行拟合和转换
            train_labels = labels_np[train_labels_idx]
            scaler.fit(train_labels.reshape(-1, 1))

            # 对所有标签进行转换
            labels_normalized = scaler.transform(labels_np.reshape(-1, 1)).flatten()

            # 将归一化后的标签转换回pandas Series
            labels = pd.Series(labels_normalized)

            # 保存scaler用于预测时的反归一化
            scaler_path = 'data/processed_data/' + f"{datasets}_{split_method}_{seed}_scaler.pth"
            torch.save(scaler, scaler_path)

        data_list = []
        failed_molecules = []  # 记录失败的分子
        for i in tqdm(range(len(smiles_list))):
            if i % 10000 == 0:
                print(i)

            smiles = smiles_list[i]

            Graph_list = path_complex_mol(smiles, encoder_atom, encoder_bond, use_cache=True, dataset_name=datasets)
            if Graph_list == False:
                failed_molecules.append(smiles)  # 记录失败的分子
                continue

            else:
                if has_node_with_zero_in_degree(Graph_list):
                    continue

                else:
                    if groups is not None:
                        data_list.append([smiles, torch.tensor(labels.iloc[i]), Graph_list, groups.iloc[i]])
                    else:
                        data_list.append([smiles, torch.tensor(labels.iloc[i]), Graph_list])

        print('Graph list was done!')
        # 保存失败分子列表
        if failed_molecules:
            failed_molecules_path = 'data/processed_data/' + f"{datasets}_{split_method}_{seed}_failed_molecules.txt"
            with open(failed_molecules_path, 'w') as f:
                for mol in failed_molecules:
                    f.write(f"{mol}\n")
            print(f"Saved {len(failed_molecules)} failed molecules to {failed_molecules_path}")

        if groups is not None:
            # 使用groups列进行划分
            train_data = []
            valid_data = []
            test_data = []

            for item in data_list:
                group = item[3] if len(item) > 3 else 'training'
                if group == 'training':
                    train_data.append([item[0], item[1], item[2]])
                elif group == 'valid':
                    valid_data.append([item[0], item[1], item[2]])
                elif group == 'test':
                    test_data.append([item[0], item[1], item[2]])
                else:
                    # 默认放到训练集
                    train_data.append([item[0], item[1], item[2]])

            print(f'Using groups-based split: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}')

        else:
            # 使用ScaffoldSplitter进行划分
            splitter = ScaffoldSplitter().split(data_list, frac_train=train_ratio, frac_valid=vali_ratio, frac_test=test_ratio)
            train_data = splitter[0]
            valid_data = splitter[1]
            test_data = splitter[2]
            print('Using scaffold-based split!')

        print('Splitter was done!')

        # 提取训练集数据
        train_label = []
        train_graph_list = []
        train_smiles_list = []
        for tmp_train_graph in train_data:
            train_label.append(tmp_train_graph[1])
            train_graph_list.append(tmp_train_graph[2])
            train_smiles_list.append(tmp_train_graph[0])

        # 提取验证集数据
        valid_label = []
        valid_graph_list = []
        valid_smiles_list = []
        for tmp_valid_graph in valid_data:
            valid_label.append(tmp_valid_graph[1])
            valid_graph_list.append(tmp_valid_graph[2])
            valid_smiles_list.append(tmp_valid_graph[0])

        # 提取测试集数据
        test_label = []
        test_graph_list = []
        test_smiles_list = []
        for tmp_test_graph in test_data:
            test_label.append(tmp_test_graph[1])
            test_graph_list.append(tmp_test_graph[2])
            test_smiles_list.append(tmp_test_graph[0])

        torch.save({
            'train_label': train_label,
            'train_graph_list': train_graph_list,
            'train_smiles_list': train_smiles_list,
            'valid_label': valid_label,
            'valid_graph_list': valid_graph_list,
            'valid_smiles_list': valid_smiles_list,
            'test_label': test_label,
            'test_graph_list': test_graph_list,
            'test_smiles_list': test_smiles_list,
            'batch_size': batch_size,
            'shuffle': True,
        }, 'data/processed_data/'+ f"{datasets}_{split_method}_{seed}" +'.pth')

        # 保存数据集缓存
        try:
            from utils.graph_path import save_all_dataset_caches
            save_all_dataset_caches()
        except:
            pass



def message_func(edges):
    return {'feat': edges.data['feat']}

def reduce_func(nodes):
    num_edges = nodes.mailbox['feat'].size(1)  
    agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  
    return {'agg_feats': agg_feats}

def update_node_features(g):
    g.send_and_recv(g.edges(), message_func, reduce_func)

    g.ndata['feat'] = torch.cat((g.ndata['feat'], g.ndata['agg_feats']), dim=1)

    return g





def train(model, device, train_loader, valid_loader, optimizer, epoch):
    model.train()
    total_train_loss = 0.0

    train_num = 0
    for batch_idx, data in enumerate(train_loader):
        train_num += len(data)
        optimizer.zero_grad()

        y = data[0].to(device)              
        g = update_node_features(data[1]).to(device) 
        x = g.ndata['feat']                

        out = model(g, x)                   

        y = y.to(dtype=out.dtype)
        mask = (y != -1).to(dtype=out.dtype)          
        y_clean = torch.where(y == -1, torch.zeros_like(y), y)

        out = out.squeeze(-1)
        loss_elem = loss_fn(out, y_clean)             
        loss = (loss_elem * mask).sum() / mask.sum().clamp_min(1.0)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()               

    model.eval()
    total_loss_val = 0.0
    valid_num = 0
    with torch.no_grad():
        for batch_idx, valid_data in enumerate(valid_loader):
            valid_num += len(valid_data)

            y = valid_data[0].to(device)
            g = update_node_features(valid_data[1]).to(device)
            x = g.ndata['feat']

            out = model(g, x)

            y = y.to(dtype=out.dtype)
            mask = (y != -1).to(dtype=out.dtype)
            y_clean = torch.where(y == -1, torch.zeros_like(y), y)

            out = out.squeeze(-1)
            loss_elem = loss_fn(out, y_clean)      
            vloss = (loss_elem * mask).sum() / mask.sum().clamp_min(1.0)

            total_loss_val += vloss.item()

    print(f"Epoch {epoch} | Train Loss: {total_train_loss / train_num :.4f} | Vali Loss: {total_loss_val / valid_num :.4f}")
    return total_train_loss, total_loss_val



"""
def train(model, device, train_loader, valid_loader, optimizer, epoch):
    model.train()

    total_train_loss = 0.0
    train_num = 0

    
    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        train_label_value = []
        y = data[0]
        train_label_value.append(torch.unsqueeze(y, dim=0))
        graph_list = update_node_features(data[1]).to(device)
        node_features = graph_list.ndata['feat'].to(device)
        #node_features = add_noise(graph_list.ndata['feat'],noise=True).to(device)
        #output = model(batch_g_list = graph_list, device = device, resent = resent,pooling=pooling).cpu()
        output = model(graph_list, node_features).cpu()
        
        arr_label = torch.Tensor().cpu()
        arr_pred = torch.Tensor().cpu()
        for j in range(y.shape[1]):
            c_valid = np.ones_like(y[:, j], dtype=bool)
            c_label, c_pred = y[c_valid, j], output[c_valid, j]
            zero = torch.zeros_like(c_label)
            c_label = torch.where(c_label == -1, zero, c_label)
            
            arr_label = torch.cat((arr_label,c_label),0)
            arr_pred = torch.cat((arr_pred,c_pred),0)
        
        arr_pred = arr_pred.float()
        arr_label = arr_label.float()

        loss = loss_fn(arr_pred, arr_label)
        #loss = FocalLoss(arr_pred, arr_label)
        train_loss = torch.sum(loss)
        total_train_loss = total_train_loss + train_loss
        train_loss.backward()
        optimizer.step()
    model.eval()
    total_loss_val = 0.0
    with torch.no_grad():
        for batch_idx, valid_data in enumerate(valid_loader):
            
            y = valid_data[0]
            train_label_value.append(torch.unsqueeze(y, dim=0))
            graph_list = update_node_features(valid_data[1]).to(device)
            node_features = graph_list.ndata['feat'].to(device)
            #node_features = add_noise(graph_list.ndata['feat'],noise=True).to(device)
            #output = model(batch_g_list = graph_list, device = device, resent = resent,pooling=pooling).cpu()
            output = model(graph_list, node_features).cpu()
            
            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)
                
                arr_label = torch.cat((arr_label,c_label),0)
                arr_pred = torch.cat((arr_pred,c_pred),0)
            
            arr_pred = arr_pred.float()
            arr_label = arr_label.float()
    
            loss = loss_fn(arr_pred, arr_label)
            valid_loss = torch.sum(loss)
            total_loss_val = total_loss_val + valid_loss
    print(f"Epoch {epoch}|Train Loss: {total_train_loss:.4f}| Vali Loss:{total_loss_val:.4f}")

    return total_train_loss, total_loss_val
"""

def predicting(model, device, data_loader, loss_type='bce', datafile=None, split_method=None, seed=None, save_pred=False, smiles_list=None, i=None):
    model.eval()

    total_preds = torch.Tensor().cpu()
    total_labels = torch.Tensor().cpu()
    all_smiles = []  # 用于收集所有的smiles

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):

            y = data[0]
            graph_list = update_node_features(data[1]).to(device)
            node_features = graph_list.ndata['feat'].to(device)

            # 获取模型输出
            output = model(graph_list, node_features).cpu()

            # 根据损失函数类型决定是否应用sigmoid
            if loss_type == 'bce':
                # 分类任务，检查模型是否已经有sigmoid层
                has_sigmoid = any(isinstance(module, nn.Sigmoid) for module in model.modules())
                if not has_sigmoid:
                    # 如果模型没有sigmoid层，则应用sigmoid（适用于BCEWithLogitsLoss）
                    output = torch.sigmoid(output)

            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()

            # 处理一维标签的情况
            if len(y.shape) == 1:
                c_valid = np.ones_like(y, dtype=bool)
                c_label, c_pred = y[c_valid], output[c_valid].squeeze(-1)
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)

                arr_label = torch.cat((arr_label,c_label),0)
                arr_pred = torch.cat((arr_pred,c_pred),0)
            else:
                # 处理多维标签的情况（多任务学习）
                for j in range(y.shape[1]):
                    c_valid = np.ones_like(y[:, j], dtype=bool)
                    c_label, c_pred = y[c_valid, j], output[c_valid, j]
                    zero = torch.zeros_like(c_label)
                    c_label = torch.where(c_label == -1, zero, c_label)

                    arr_label = torch.cat((arr_label,c_label),0)
                    arr_pred = torch.cat((arr_pred,c_pred),0)

            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)

    # 对回归任务的预测结果进行反归一化
    if loss_type != 'bce' and datafile is not None and split_method is not None and seed is not None:
        # 检查是否存在scaler文件
        scaler_path = 'data/processed_data/' + f"{datafile}_{split_method}_{seed}_scaler.pth"
        if os.path.exists(scaler_path):
            # 加载scaler
            scaler = torch.load(scaler_path)
            # 对预测结果和标签进行反归一化
            total_preds_np = total_preds.numpy().flatten()
            total_labels_np = total_labels.numpy().flatten()

            # 反归一化
            total_preds_denormalized = scaler.inverse_transform(total_preds_np.reshape(-1, 1)).flatten()
            total_labels_denormalized = scaler.inverse_transform(total_labels_np.reshape(-1, 1)).flatten()

            # 转换回tensor
            total_preds = torch.tensor(total_preds_denormalized)
            total_labels = torch.tensor(total_labels_denormalized)

    if save_pred and smiles_list is not None:
        print(f"smiles: {len(smiles_list)}; pred: {len(total_preds)}; label: {len(total_labels)}")
        df = pd.DataFrame({'smiles': smiles_list, 'pred': total_preds.numpy().flatten(), 'label': total_labels.numpy().flatten()})
        # 保存预测结果到CSV文件
        pred_file_path = 'predictions/' + f"KA-GNN_{datafile}_{split_method}_{seed}_{i}_test_prediction.csv"
        df.to_csv(pred_file_path, index=False)
        print(f"Predictions saved to {pred_file_path}")

    # 根据损失函数类型选择评估指标
    if loss_type == 'bce':
        # 分类任务使用AUC评估
        metric1 = roc_auc_score(total_labels.numpy().flatten(), total_preds.numpy().flatten())
        metric2 = accuracy_score(total_labels.numpy().flatten(), torch.round(total_preds).numpy().flatten())
        precision, recall, _ = precision_recall_curve(total_labels.numpy().flatten(), total_preds.numpy().flatten())
        metric3 = auc(recall, precision)
    else:
        # 回归任务使用R²评估
        metric1 = r2_score(total_labels.numpy().flatten(), total_preds.numpy().flatten())
        metric2 = np.sqrt(mean_squared_error(total_labels.numpy().flatten(), total_preds.numpy().flatten()))
        metric3 = mean_absolute_error(total_labels.numpy().flatten(), total_preds.numpy().flatten())

    return metric1, metric2, metric3



def parse_arguments():
    parser = argparse.ArgumentParser(description="help")


    parser.add_argument("--config", type=str, help="path")
    parser.add_argument("--select_dataset", type=str, required=True, help="select dataset")
    parser.add_argument("--loss_sclect", type=str, required=True, help="loss type")
    parser.add_argument("--split_method", type=str, required=True, help="split method")
    parser.add_argument("--seed", type=int, required=True, help="seed")

    args = parser.parse_args()
    args.config = './config/c_path.yaml'
    if args.config:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
        for key, value in config.items():
            setattr(args, key, value)

    return args


if __name__ == '__main__':
    
    #mp.set_start_method('spawn', force=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    args = parse_arguments()
    for key, value in vars(args).items():
        if key != 'config':
            print(f"{key}: {value}")
    datafile = args.select_dataset
    split_method = args.split_method
    seed = args.seed
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    vali_ratio = args.vali_ratio
    test_ratio = args.test_ratio
    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
    target_dim = 1

    

    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond

    encode_dim = [0,0]
    encode_dim[0] = 92
    encode_dim[1] = 21
    

    
    creat_data(datafile, split_method, seed, encoder_atom, encoder_bond, batch_size, train_ratio, vali_ratio, test_ratio, loss_type=args.loss_sclect)

    model_select = args.model_select
    loss_sclect = args.loss_sclect

    state = torch.load('data/processed_data/'+ f"{datafile}_{split_method}_{seed}" +'.pth')

    loaded_train_dataset = CustomDataset(state['train_label'], state['train_graph_list'])
    loaded_valid_dataset = CustomDataset(state['valid_label'], state['valid_graph_list'])
    loaded_test_dataset = CustomDataset(state['test_label'], state['test_graph_list'])

    # 提取smiles_list
    train_smiles_list = state.get('train_smiles_list', [])
    valid_smiles_list = state.get('valid_smiles_list', [])
    test_smiles_list = state.get('test_smiles_list', [])

    # 训练时使用shuffle，预测时不使用shuffle以保持smiles_list顺序一致
    loaded_train_loader = DataLoader(loaded_train_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    if vali_ratio == 0.0:
        loaded_valid_loader = []
    else:
        loaded_valid_loader = DataLoader(loaded_valid_dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    loaded_test_loader = DataLoader(loaded_test_dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_fn)


    print('dataset was loaded!')

    print("length of training set:",len(loaded_train_dataset))
    print("length of validation set:",len(loaded_valid_dataset))
    print("length of testing set:",len(loaded_test_dataset))
    
    iter = args.iter
    LR = args.LR
    NUM_EPOCHS = args.NUM_EPOCHS
    grid_feat = args.grid_feat
    num_layers = args.num_layers
    pooling = args.pooling

    All_AUC = []

    start_time = time.time()

    if args.loss_sclect == 'bce':
        result_pd = pd.DataFrame(columns=['roc_auc', 'roc_prc', 'accuracy'])
    else:
        result_pd = pd.DataFrame(columns=['r2', 'rmse', 'mae'])
    for i in range(iter):

        # 设置种子
        SEED = 2024 + 10 * i
        set_seed(SEED)
        
        if model_select == 'ka_gnn':
            model = KA_GNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'ka_gnn_two':
            model = KA_GNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                               grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)
        
        elif model_select == 'mlp_sage':
            model = MLPGNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'mlp_sage_two':
            model = MLPGNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                               grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'kan_sage':
            model = KANGNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'kan_sage_two':
            model = KANGNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        train_loss_dic = {}
        vali_loss_dic = {}

        model = model.to(device)
        if loss_sclect == 'l1':
            #loss_fn = nn.L1Loss()
            loss_fn = nn.L1Loss(reduction='sum')#sum，mean,none

        elif loss_sclect == 'l2':
            loss_fn = nn.MSELoss(reduction='none')

        elif loss_sclect == 'sml1':
            loss_fn = nn.SmoothL1Loss(reduction='sum')#mean,none,sum

        elif loss_sclect == 'bce':
            # loss_fn = nn.BCELoss(reduction='mean')
            loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        
        else:
            print('No Found the Loss function!')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        best_epoch = 0
        best_metric = -float('inf')
        for epoch in range(NUM_EPOCHS):
            train_loss,vali_loss = train(model, device, loaded_train_loader, loaded_valid_loader, optimizer, epoch + 1)
            metric1, metric2, metric3 = predicting(model, device, loaded_valid_loader, loss_type=loss_sclect, datafile=datafile, split_method=split_method, seed=seed)
            if metric1 > best_metric:
                best_metric = metric1
                best_epoch = epoch + 1
                logger.info(f'Metric: {best_metric:.5f}')
                formatted_number = "{:.5f}".format(best_metric)
                best_metric = float(formatted_number)
                print(f"Epoch [{epoch+1}], Learning Rate: {scheduler.get_last_lr()}")
                torch.save(model.state_dict(), f'./model_weights/{datafile}_{split_method}_{seed}_{i+1}.pth')
            else:
                print(f"out of patience {epoch + 1 - best_epoch}")
        
            if epoch % 10 == 0:
                #MAE_list.append(best_MAE)
                print("-------------------------------------------------------")
                print("epoch:",epoch)
                print('best_metric:', best_metric)

            if epoch == NUM_EPOCHS-1:
                print(f"the best result up to {i+1}-loop is {best_metric:.4f}.")
                formatted_number = "{:.5f}".format(best_metric)
                All_AUC.append(best_metric)
            
            if epoch + 1 - best_epoch >= 30:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        model.load_state_dict(torch.load(f'./model_weights/{datafile}_{split_method}_{seed}_{i+1}.pth'))
        model.to(device)
        train_metric1, train_metric2, train_metric3 = predicting(model, device, loaded_train_loader, loss_type=loss_sclect, datafile=datafile,
                                               split_method=split_method, seed=seed, smiles_list=train_smiles_list,
                                               save_pred=False, i=i+1)
        valid_metric1, valid_metric2, valid_metric3 = predicting(model, device, loaded_valid_loader, loss_type=loss_sclect, datafile=datafile,
                                               split_method=split_method, seed=seed, smiles_list=valid_smiles_list,
                                               save_pred=False, i=i+1)
        test_metric1, test_metric2, test_metric3 = predicting(model, device, loaded_test_loader, loss_type=loss_sclect, datafile=datafile,
                                               split_method=split_method, seed=seed, smiles_list=test_smiles_list,
                                               save_pred=True, i=i+1)
        
        result_pd['train_' + str(i + 1)] = [train_metric1, train_metric2, train_metric3]
        result_pd['val_' + str(i + 1)] = [valid_metric1, valid_metric2, valid_metric3]
        result_pd['test_' + str(i + 1)] = [test_metric1, test_metric2, test_metric3]

    result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
    result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
    result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
    result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
    result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
    result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
    os.makedirs('./result/', exist_ok=True)
    result_pd.to_csv('./result/KA-GNN_' + args.select_dataset + '_' + args.split_method + '_' + str(args.seed) + '_all_result.csv', index=False)

