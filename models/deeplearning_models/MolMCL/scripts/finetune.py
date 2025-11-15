import os
import sys
import pickle
import math
import copy
import random
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
import argparse
import yaml
from tqdm import tqdm
from collections import Counter
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_dense_batch
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import time

script_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(script_dir) 

sys.path.append(project_root)

from molmcl.finetune.loader import MoleculeDataset
from molmcl.finetune.model import GNNPredictor
from molmcl.finetune.prompt_optim import optimize_prompt_weight_ri as optimize_prompt_weight_ri_
from molmcl.splitters import scaffold_split, moleculeace_split, random_split
from molmcl.utils.scheduler import PolynomialDecayLR


def get_optimizer(model, lr_params):
    assert isinstance(lr_params, dict)

    pretrain_name, prompt_name, finetune_name = [], [], []
    for name, param in model.named_parameters():
        if 'gnn' in name or 'aggr' in name:
            pretrain_name.append(name)
        elif 'graph_pred_linear' in name:
            finetune_name.append(name)
        else:
            prompt_name.append(name)

    pretrain_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in pretrain_name, model.named_parameters()))))
    finetune_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in finetune_name, model.named_parameters()))))
    prompt_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in prompt_name, model.named_parameters()))))

    # Adam, (Adadelta), Adagrad, RAdam
    optimizer = torch.optim.AdamW([
        {'params': finetune_params},
        {'params': pretrain_params, 'lr': float(lr_params['pretrain_lr'])},
        {'params': prompt_params, 'lr': float(lr_params['prompt_lr'])}
    ], lr=float(lr_params['finetune_lr']), weight_decay=float(lr_params['decay']))

    return optimizer


def get_dataloader(config, seed=0):
    # Setup dataset
    dataset = MoleculeDataset(config['dataset']['data_dir'],
                              config['dataset']['data_name'],
                              config['dataset']['feat_type'],
                              config['dataset']['split'],
                              config['dataset']['task'])

    num_task = dataset.num_task
    print('Loading dataset {} of size {} with num_task={}'.format(config['dataset']['data_name'], len(dataset), num_task))

    if 'ADMET' in config['dataset']['data_dir']:
        splits = np.array(dataset.splits)
        train_idx = np.argwhere(splits == 'training').flatten()
        val_idx = np.argwhere(splits == 'valid').flatten()
        test_idx = np.argwhere(splits == 'test').flatten()
    elif 'CHEMBL' in config['dataset']['data_name']:  # MoleculeACE stratified random split
        train_idx, val_idx, test_idx = moleculeace_split(dataset.smiles, dataset.labels, val_size=0.1, test_size=0.1)
    else:  # MoleculeNet scaffold split
        train_idx, val_idx, test_idx = scaffold_split(dataset.smiles, frac_valid=0.1, frac_test=0.1, balanced=False)

    if config['dataset']['task'] == 'classification':
        label_counter = Counter([dataset.labels[i][0] for i in train_idx])
        pos_weight = label_counter[-1] / label_counter[1]
    else:
        pos_weight = None

    train_dataset, val_dataset, test_dataset = \
        Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return dataset, train_loader, val_loader, test_loader, pos_weight


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def warmup_cosine_scheduler(optimizer, warmup_updates, tot_updates, base_lr, end_lr=0.0):
    def lr_lambda(update):
        # 1) Warmup phase: linearly scale from 0 -> base_lr
        if update < warmup_updates:
            return float(update) / float(max(1, warmup_updates))

        # 2) Cosine annealing phase
        progress = float(update - warmup_updates) / float(max(1, tot_updates - warmup_updates))
        # Cosine from 1 (at start) down to 0 (at end)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale range from base_lr -> end_lr
        # Start of cosine: 1.0 -> base_lr, end of cosine: 0.0 -> end_lr
        return cosine * (1.0 - end_lr / base_lr) + end_lr / base_lr

    return LambdaLR(optimizer, lr_lambda)


def eval(model, val_loader, config, metric='r2', return_result=False):
    assert metric in ['rmse', 'r2']
    model.eval()
    y_true, y_scores, y_smi = [], [], []
    for step, batch in enumerate(val_loader):
        batch = batch.to(config['device'])
        with torch.no_grad():
            predict = model(batch)['predict']

        y_true.append(batch.label.view(predict.shape))
        y_scores.append(predict)
        y_smi += batch['smi']

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if config['dataset']['task'] == 'regression':
        if metric == 'rmse':
            score = -mean_squared_error(y_true, y_scores, squared=False)
        else:
            score = r2_score(y_true, y_scores)
    else:
        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i] ** 2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

        score = np.mean(roc_list)

        y_true = (y_true + 1) / 2
        y_scores = sigmoid(y_scores)

    if not return_result:
        return score
    else:
        return score, (y_smi, y_true.flatten(), y_scores.flatten())


def train(model, train_loader, criterion, optimizer, scheduler, config, channel_idx=-1):
    model.train()
    loss_history = []
    channel_weight = 0
    for idx, batch in enumerate(train_loader):
        batch.to(config['device'])
        output = model(batch, channel_idx=channel_idx)
        predict = output['predict']
        label = batch.label.view(predict.shape)

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            mask = label == 0  # nan entry
            loss = criterion(predict.double(), (label + 1) / 2) * (~mask)
            loss = loss.sum() / (~mask).sum()
        elif isinstance(criterion, nn.MSELoss):
            loss = criterion(predict, label)
            loss = loss.mean()
        else:
            raise Exception

        optimizer.zero_grad()
        loss.backward()
        if config['optim']['gradient_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['optim']['gradient_clip'])
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_history.append(loss.item())

    channel_weight = channel_weight / len(train_loader)

    return np.mean(loss_history), channel_weight


def optimize_prompt_weight_ri(model, train_loader, val_loader, config, metric='euclidean', act='softmax', max_num=5000):
    assert metric in ['euclidean', 'cosine']

    temperature = config['model']['temperature']
    skip_bo = config['prompt_optim']['skip_bo']

    # Extract channel-wise embeddings for all training data
    num = 0
    model.eval()
    graph_rep_list, label_list = [], []
    for loader in [train_loader, val_loader]:
        if loader is None:
            continue
        for batch in loader:
            batch.to(config['device'])
            with torch.no_grad():
                graph_reps = []
                if model.backbone == 'gps':
                    h_g, node_repres = model.gnn(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    h_g, node_repres = model.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                # map back to batched nodes for aggregation
                batch_x, batch_mask = to_dense_batch(node_repres, batch.batch)

                # conditional aggregation given the prompt_inds
                for i in range(len(model.prompt_token)):
                    h_g, h_x, _ = model.aggrs[i](batch_x, batch_mask)
                    if config['model']['normalize']:
                        h_g = F.normalize(h_g, dim=-1)
                    graph_reps.append(h_g)

            graph_reps_batch = torch.stack(graph_reps)
            labels_batch = batch.label.view(-1, model.num_tasks)

            is_valid = (labels_batch != 0).sum(-1) == labels_batch.size(1)
            graph_rep_list.append(graph_reps_batch[:, is_valid])
            label_list.append(labels_batch[is_valid])

            num += graph_rep_list[-1].size(1)
            if num > max_num:
                break

    graph_reps = torch.concat(graph_rep_list, dim=1).cpu()  # (num_prompt, N, emb_dim)
    labels = torch.concat(label_list, dim=0).cpu()  # (N, 1)

    return optimize_prompt_weight_ri_(graph_reps, labels, n_runs=50, n_inits=100, n_points=5, n_restarts=512,
                                      n_samples=512, temperature=temperature, metric=metric,
                                      act=act, skip_bo=skip_bo, verbose=config['verbose'])


def main(config):
    if not config['model']['checkpoint']:
        config['model']['use_prompt'] = False

    if not os.path.exists(f"./{config['save_dir']}"):
        os.mkdir(f"./{config['save_dir']}")

    # runseeds = np.random.randint(100, size=config['num_run'])
    runseeds = [2024, 2034, 2044, 2054, 2064]

    # Setup model
    if config['dataset']['feat_type'] == 'basic':
        atom_feat_dim, bond_feat_dim = None, None
    elif config['dataset']['feat_type'] == 'rich':
        atom_feat_dim, bond_feat_dim = 143, 14
    elif config['dataset']['feat_type'] == 'super_rich':
        atom_feat_dim, bond_feat_dim = 170, 14
    else:
        raise NotImplementedError('Unrecognized feature type. Please choose from [basic/rich/super_rich].')

    # Setup dataset and dataloader
    dataset, train_loader, val_loader, test_loader, pos_weight = get_dataloader(config)

    # Main:
    avg_auc_last, avg_auc_best = [], []

    best_initialization = None
    if config['prompt_optim']['inits']:
        best_initialization = torch.Tensor(config['prompt_optim']['inits'])

    
    for i in range(config['num_run']):
        # Setup model
        model = GNNPredictor(num_layer=config['model']['num_layer'],
                             emb_dim=config['model']['emb_dim'],
                             num_tasks=dataset.num_task,
                             normalize=config['model']['normalize'],
                             atom_feat_dim=atom_feat_dim,
                             bond_feat_dim=bond_feat_dim,
                             drop_ratio=config['model']['dropout_ratio'],
                             attn_drop_ratio=config['model']['attn_dropout_ratio'],
                             temperature=config['model']['temperature'],
                             use_prompt=config['model']['use_prompt'],
                             model_head=config['model']['heads'],
                             layer_norm_out=config['model']['layernorm'], 
                             backbone=config['model']['backbone'],
                             act='softmax',
                             use_descriptor=config['model']['use_descriptor'],
                             use_fingerprint=config['model']['use_fingerprint'],)

        if config['model']['checkpoint']:
            print('Loading checkpoint from {}'.format(config['model']['checkpoint']))
            model.load_state_dict(torch.load(config['model']['checkpoint'])['wrapper'], strict=False)
        model.to(config['device'])

        # Train prompt:
        if config['model']['use_prompt']:
            if best_initialization is None:
                set_seed(0)
                best_initialization = optimize_prompt_weight_ri(model, train_loader, None, config,
                                                                metric='euclidean', act='softmax')
            model.set_prompt_weight(best_initialization.to(config['device']))

            initial_prompt_weights = model.get_prompt_weight('none').data.cpu()
            initial_prompt_probs = model.get_prompt_weight(model.act).data.cpu()
            print('Initial prompt weight:', initial_prompt_weights)
            print('Initial prompt prob:  ', initial_prompt_probs)

        # Setup optimizer
        optimizer = get_optimizer(model, config['optim'])
        scheduler = None
        if config['optim']['scheduler'] == 'cos_anneal':
            total_updates = config['epochs'] * len(train_loader)
            if config['optim']['warmup'] > 0:
                scheduler = warmup_cosine_scheduler(optimizer,
                                                    warmup_updates=int(total_updates * config['optim']['warmup']),
                                                    tot_updates=total_updates,
                                                    base_lr=config['optim']['finetune_lr'], end_lr=1e-5)
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'] * len(train_loader), eta_min=1e-5)

        elif config['optim']['scheduler'] == 'poly_decay':
            total_updates = config['epochs'] * len(train_loader)
            scheduler = PolynomialDecayLR(optimizer,
                                          warmup_updates=int(total_updates * config['optim']['warmup']),
                                          tot_updates=total_updates,
                                          lr=config['optim']['finetune_lr'], end_lr=1e-5, power=1)

        # Setup loss function
        if config['dataset']['task'] == 'regression':
            criterion = nn.MSELoss(reduction='none')
        elif config['dataset']['task'] == 'classification':
            # criterion = nn.BCEWithLogitsLoss(reduction='none')
            criterion = nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=torch.Tensor([pos_weight]).to(config['device']))
        else:
            raise NotImplementedError

        best_score, best_checkpoint = -float('inf'), None

        # Setup learnable parameters:
        if not config['model']['update_aggr']:
            model.freeze_aggr_module()

        # Setup random seed
        print("Seed:", runseeds[i])
        set_seed(runseeds[i])

        # Save pretrain representation (only at run #1)
        if i == 0:
            model.eval()
            split2rep = {}
            splits = ['train', 'val', 'test']
            for s_i, loader in enumerate([train_loader, val_loader, test_loader]):
                mol_smiles, mol_reps = [], []
                for batch in loader:
                    batch.to(config['device'])
                    with torch.no_grad():
                        graph_reps = model.get_representations(batch)
                        if config['model']['use_prompt']:
                            prompt_weight = model.get_prompt_weight(act=model.act)
                            graph_rep = torch.matmul(graph_reps.transpose(0, 2), prompt_weight).transpose(0, 1)
                        else:
                            prompt_weight = None
                            graph_rep = graph_reps

                    mol_smiles += batch['smi']
                    mol_reps.append(graph_rep)

                mol_reps = torch.cat(mol_reps, dim=0).cpu().numpy()
                split2rep[splits[s_i]] = (mol_smiles, mol_reps)

            save_path = f'./{config["save_dir"]}/{config["dataset"]["data_name"]}_{config["dataset"]["split"]}_rep.pk'
            with open(save_path, 'wb') as f:
                pickle.dump((prompt_weight, split2rep), f)

        # Start fine-tuning
        model.train()
        max_patience = config['optim']['patience'] if config['optim']['patience'] > 0 else float('inf')
        num_patience = 0
        for epoch in tqdm(range(1, config['epochs'] + 1)):
            # train one epoch
            train(model, train_loader, criterion, optimizer, scheduler, config)

            # evaluate
            score = eval(model, val_loader, config, return_result=False)

            if config['verbose'] and config['model']['use_prompt']:
                # train_score = eval(model, val_loader, config, return_result=False)
                test_score = eval(model, test_loader, config, return_result=False)
                weight = model.get_prompt_weight(model.act).data.cpu().numpy()
                cur_lr = optimizer.param_groups[-1]['lr']
                tqdm.write(
                    f"[ep{epoch}] {score:>4.4f} {test_score:>4.4f} {cur_lr} [{weight[0]:>4.3f} {weight[1]:>4.3f} {weight[2]:>4.3f}]")
            elif config['verbose']:
                # train_score = eval(model, val_loader, config, return_result=False)
                test_score = eval(model, test_loader, config, return_result=False)
                cur_lr = optimizer.param_groups[-1]['lr']
                tqdm.write(f"[ep{epoch}] {score:>4.4f} {test_score:>4.4f} {cur_lr}")

            if score > best_score:
                best_score = score
                best_checkpoint = copy.deepcopy(model.state_dict())
                num_patience = 0
            else:
                num_patience += 1

            if num_patience >= max_patience:
                break

        score_last_checkpoint = eval(model, test_loader, config)
        avg_auc_last.append(score_last_checkpoint)
        if config['model']['use_prompt']:
            print('Prompt weight of last checkpoint:', model.get_prompt_weight(model.act).data.cpu())

        model.load_state_dict(best_checkpoint)

        # Save train, val, test predictions from best validated checkpoint
        split2res = {}
        train_score_best_checkpoint, train_result = eval(model, train_loader, config, return_result=True)
        val_score_best_checkpoint, val_result = eval(model, val_loader, config, return_result=True)
        score_best_checkpoint, test_result = eval(model, test_loader, config, return_result=True)
        split2res['train'] = (train_score_best_checkpoint, train_result)
        split2res['val'] = (val_score_best_checkpoint, val_result)
        split2res['test'] = (score_best_checkpoint, test_result)
        save_path = f'./{config["save_dir"]}/{config["dataset"]["data_name"]}_{config["dataset"]["split"]}_run_{i+1}.pk'
        with open(save_path, 'wb') as f:
            pickle.dump(split2res, f)

        avg_auc_best.append(score_best_checkpoint)
        if config['model']['use_prompt']:
            print('Prompt weight of best checkpoint:', model.get_prompt_weight(model.act).data.cpu())

        if 'CHEMBL' in config['dataset']['data_name']:
            print('[Best R2]: {:.4f} {:.4f} {:.4f}'.format(best_score, score_last_checkpoint, score_best_checkpoint))
        else:
            print('[Best AUC]: {:.4f} {:.4f} {:.4f}'.format(best_score, score_last_checkpoint, score_best_checkpoint))

    print(avg_auc_last)
    print('[Last] {} {}'.format(np.mean(avg_auc_last), np.std(avg_auc_last)))
    print(avg_auc_best)
    print('[Best] {} {}'.format(np.mean(avg_auc_best), np.std(avg_auc_best)))

    with open(f"./{config['save_dir']}/{config['dataset']['data_name']}_{config['dataset']['split']}.txt", 'w') as f:
        for i in range(len(avg_auc_best)):
            f.write('Run #{} (seed={}): best={} last={}\n'.format(i + 1, runseeds[i], avg_auc_best[i], avg_auc_last[i]))
        f.write('Average last score: {}\n'.format(np.mean(avg_auc_last)))
        f.write('Average best score: {}\n'.format(np.mean(avg_auc_best)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_type', type=str)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data_name', type=str, default='BBBP')
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open('./config/ADMET.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    config['dataset']['data_name'] = args.data_name
    config['dataset']['split'] = args.split_type

    sub_folder = f"data_with_group_{args.split_type.split('_')[0]}"
    config['dataset']['data_dir'] = os.path.join(config['dataset']['data_dir'], sub_folder)

    if config['dataset']['data_name'] in ['BBBP', 'HLM_metabolic_stability', 'Mutagenicity', 'oral_bioavailability', 'hERG']:
        config['dataset']['task'] = 'classification'
    else:
        config['dataset']['task'] = 'regression'

    # if 'ADMET' in sys.argv[1]:
    #     with open('./config/ADMET.yaml', 'r') as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)
    #     config['dataset']['data_name'] = sys.argv[1].split('/')[1]
    # elif 'CHEMBL' in sys.argv[1]:
    #     if config_file:
    #         with open(config_file, 'r') as f:
    #             config = yaml.load(f, Loader=yaml.FullLoader)
    #     else:
    #         with open('./config/moleculeace/chembl.yaml', 'r') as f:
    #             config = yaml.load(f, Loader=yaml.FullLoader)
    #     config['dataset']['data_name'] = sys.argv[1].split('/')[1]
    # else:
    #     with open('./config/{}.yaml'.format(sys.argv[1]), 'r') as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)
    main(config)
