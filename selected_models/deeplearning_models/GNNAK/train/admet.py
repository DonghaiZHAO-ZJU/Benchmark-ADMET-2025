
import torch
from core.config import cfg, update_cfg
from core.train_helper import run, run_validation
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform

from core.data import calculate_stats
from core.data import mydata
from train.train_utils.evaluator import Meter

import os
import pandas as pd

def standardization_np(data, mean, std):
    return (data - mean) / (std + 1e-10)
def re_standar_np(data, mean, std):
    return data * (std + 1e-10) + mean

def create_dataset(cfg): 
    torch.set_num_threads(cfg.num_workers)
    # No need to do offline transformation
    transform = SubgraphsTransform(cfg.subgraph.hops, 
                                   walk_length=cfg.subgraph.walk_length, 
                                   p=cfg.subgraph.walk_p, 
                                   q=cfg.subgraph.walk_q, 
                                   repeat=cfg.subgraph.walk_repeat,
                                   sampling_mode=cfg.sampling.mode, 
                                   minimum_redundancy=cfg.sampling.redundancy, 
                                   shortest_path_mode_stride=cfg.sampling.stride, 
                                   random_mode_sampling_rate=cfg.sampling.random_rate,
                                   random_init=True)

    transform_eval = SubgraphsTransform(cfg.subgraph.hops, 
                                        walk_length=cfg.subgraph.walk_length, 
                                        p=cfg.subgraph.walk_p, 
                                        q=cfg.subgraph.walk_q, 
                                        repeat=cfg.subgraph.walk_repeat,
                                        sampling_mode=None, 
                                        random_init=False)
    root = '../data/admet'
    train_dataset = mydata(root=root, data_name=cfg.dataset_name, split='training', transform=transform)
    if cfg.dataset_type=='classification':
        mean, std = None, None
    else:
        mean, std = train_dataset.get_scaler()
    print(f'mean: {mean}, std: {std}')
    valid_dataset = mydata(root=root, data_name=cfg.dataset_name, split='valid', transform=transform_eval)
    test_dataset = mydata(root=root, data_name=cfg.dataset_name, split='test', transform=transform_eval)

    cfg.nfeat_node = train_dataset.data.x.size(-1)
    cfg.nfeat_edge = train_dataset.data.edge_attr.size(-1)
    cfg.label_weight = train_dataset.label_weight()

    if (cfg.sampling.mode is None and cfg.subgraph.walk_length == 0) or (cfg.subgraph.online is False):
        train_dataset = [x for x in train_dataset]
    valid_dataset = [x for x in valid_dataset] 
    test_dataset = [x for x in test_dataset]

    return train_dataset, valid_dataset, test_dataset, mean, std

def create_model(cfg):
    model = GNNAsKernel(cfg.nfeat_node, cfg.nfeat_edge, 
                        nhid=cfg.model.hidden_size, 
                        nout=1, 
                        nlayer_outer=cfg.model.num_layers,
                        nlayer_inner=cfg.model.mini_layers,
                        gnn_types=[cfg.model.gnn_type], 
                        hop_dim=cfg.model.hops_dim,
                        use_normal_gnn=cfg.model.use_normal_gnn, 
                        vn=cfg.model.virtual_node, 
                        pooling=cfg.model.pool,
                        embs=cfg.model.embs,
                        embs_combine_mode=cfg.model.embs_combine_mode,
                        mlp_layers=cfg.model.mlp_layers,
                        dropout=cfg.train.dropout, 
                        subsampling=True if cfg.sampling.mode is not None else False,
                        online=cfg.subgraph.online) 
    return model

def train(train_loader, model, optimizer, mean, std, device):
    total_loss = 0
    N = 0
    if cfg.dataset_type=='classification':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.label_weight]).to(device))
    else:
        criterion = torch.nn.MSELoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = torch.unsqueeze(data.y, dim=1)
        if (mean is not None) and (std is not None):
            y = standardization_np(y, mean, std)
        loss = criterion(out, y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()
    return total_loss / N

@torch.no_grad()
def test(loader, model, mean, std, evaluator, device, in_path=None, out_path=None):
    for data in loader:
        data = data.to(device)
        y_preds = model(data)
        if (cfg.dataset_type=='regression') and (mean is not None) and (std is not None):
            y_preds = re_standar_np(y_preds, mean, std)
        y_trues = torch.unsqueeze(data.y, dim=1)
        evaluator.update(y_preds, y_trues)
    if cfg.dataset_type=='classification':
        roc_auc = evaluator.compute_metric('roc_auc')[0]
        roc_prc = evaluator.compute_metric('roc_prc')[0]
        acc = evaluator.compute_metric('acc')[0]
        if in_path and out_path:
            pred, true = evaluator.compute_metric('return_pred_true')
            pred = torch.sigmoid(pred)
            data_origin = pd.read_csv(in_path)
            data = pd.DataFrame({"smiles": data_origin[data_origin['group']=="test"]['smiles'].values, "pred": pred.flatten(), "label": true.flatten()})
            os.makedirs('./prediction', exist_ok=True)
            data.to_csv(out_path, index=False)
        evaluator.reset()
        return [roc_auc, roc_prc, acc]
    else:
        r2 = evaluator.compute_metric('r2')[0]
        rmse = evaluator.compute_metric('rmse')[0]
        mae = evaluator.compute_metric('mae')[0]
        if in_path and out_path:
            pred, true = evaluator.compute_metric('return_pred_true')
            data_origin = pd.read_csv(in_path)
            data = pd.DataFrame({"smiles": data_origin[data_origin['group']=="test"]['smiles'].values, "pred": pred.flatten(), "label": true.flatten()})
            os.makedirs('./prediction', exist_ok=True)
            data.to_csv(out_path, index=False)
        evaluator.reset()
        return [r2, rmse, mae]

if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/configs/admet.yaml')
    cfg = update_cfg(cfg)
    evaluator = Meter()
    run(cfg, create_dataset, create_model, train, test, evaluator=evaluator)
    # run_validation(cfg, create_dataset, create_model, test, evaluator=evaluator)