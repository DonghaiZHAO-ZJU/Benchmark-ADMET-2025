import os
import time
import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from create_data import data_set, collate_fn
from DNN_utils.trainer import train_an_epoch, eval_an_epoch, EarlyStopping, set_random_seed
from DNN_utils.DNN import mlp

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--fp_name", type=str, required=True)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--scaler_name", type=str, default='StandardScaler')
parser.add_argument("--use_scaler", action="store_true")
parser.add_argument("--task_type", type=str, choices=['classification', 'regression'])
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--mode", type=str, choices=['higher', 'lower'], default='higher')
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--layer_num", type=int, default=3)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.2)
args = parser.parse_args()
print("arguments\t", args, flush=True)


result_pd = pd.DataFrame()
if args.task_type=='classification':
    result_pd['index'] = ['roc_auc', 'roc_prc', 'accuracy']
else:
    result_pd['index'] = ['r2', 'rmse', 'mae']
start = time.time()
for i in range(5):
    seed=args.seed+i*10
    print("seed: ",seed, flush=True)
    set_random_seed(seed)

    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    train_data = data_set(args.data_name, args.fp_name, data_type='training')
    valid_data = data_set(args.data_name, args.fp_name, data_type='valid')
    test_data = data_set(args.data_name, args.fp_name, data_type='test')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn)

    if args.task_type=='classification':
        loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=train_data.get_label_weight().to(device))
        metric_name = 'roc_auc'
        scaler = None
    else:
        loss_criterion = torch.nn.MSELoss(reduction='none')
        if args.use_scaler:
            scaler = train_data.data_standard(args.scaler_name)
        else:
            scaler = None
        metric_name = 'r2'


    model = mlp(input_dim=train_data.fp_length, layer_num=args.layer_num, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    os.makedirs(f'./model/{args.fp_name}/', exist_ok=True)
    stopper = EarlyStopping(mode=args.mode, patience=args.patience, filename=f'./model/{args.fp_name}/{args.data_name}_{i+1}.pth')
    for e in range(args.epochs):
        loss = train_an_epoch(args, device, train_loader, model, loss_criterion, optimizer, scaler)
        train_score = eval_an_epoch(args, device, train_loader, model, scaler)
        valid_score = eval_an_epoch(args, device, valid_loader, model, scaler)
        test_score = eval_an_epoch(args, device, test_loader, model, scaler)
        early_stop = stopper.step(valid_score[0], model)
        print('epoch {:d}/{:d}, {}, lr: {:.6f}, train_loss: {:.4f}, train: {:.4f}, valid: {:.4f}, best valid score {:.4f}, test: {:.4f}'.
            format(e+1, args.epochs, metric_name, args.lr, loss, train_score[0], valid_score[0], stopper.best_score, test_score[0]), flush=True)
        if early_stop:
            break
    stopper.load_checkpoint(model)
    train_score = eval_an_epoch(args, device, train_loader, model, scaler)
    valid_score = eval_an_epoch(args, device, valid_loader, model, scaler)
    os.makedirs(f'./prediction/{args.fp_name}/', exist_ok=True)
    test_score = eval_an_epoch(args, device, test_loader, model, scaler, out_path=f'./prediction/{args.fp_name}/{args.data_name}_{i+1}.csv')

    result_pd['train_' + str(i+1)] = train_score
    result_pd['val_' + str(i+1)] = valid_score
    result_pd['test_' + str(i+1)] = test_score
    print(f"train {metric_name}: {train_score[0]:.4f}; valid {metric_name}: {valid_score[0]:.4f}; test {metric_name}: {test_score[0]:.4f}", flush=True)

elapsed = (time.time() - start)
m, s = divmod(elapsed, 60)
h, m = divmod(m, 60)
print("{} {} time used: {:d}:{:d}:{:d}".format(args.fp_name, args.data_name, int(h), int(m), int(s)), flush=True)

result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
os.makedirs(f'./result/{args.fp_name}/', exist_ok=True)
result_pd.to_csv(f'./result/{args.fp_name}/DNN_{args.fp_name}_{args.data_name}_all_result.csv', index=False)