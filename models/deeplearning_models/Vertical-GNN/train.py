from utils import seed_everything, LoadDataset
from config import NUM_FEATURES, NUM_GRAPHS_PER_BATCH, NUM_TARGET, EDGE_DIM, DEVICE, EPOCHS, params_vertical_gnn
from engine import Engine_class, Engine_regression
from model import VerticalGNN

import torch
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import os 

import argparse
import time


def run_training(train_loader, valid_loader, test_loader, model, task_type, model_path, mean, std):
    if task_type == 'classification':
        eng = Engine_class(model, optimizer, device=DEVICE)
        best_score = 0
    else:
        eng = Engine_regression(model, optimizer, mean, std, device=DEVICE, )
        best_score = -np.inf

    early_stopping_iter = PATIENCE
    early_stopping_counter = 0 

    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        train_score = eng.validate(train_loader)[0]
        valid_score = eng.validate(valid_loader)[0]
        test_score = eng.validate(test_loader)[0]
        print(f'Epoch: {epoch+1}/{EPOCHS}, train loss: {train_loss}, train score: {train_score}, valid score: {valid_score}, test score: {test_score}')
        if valid_score > best_score:
            best_score = valid_score
            early_stopping_counter=0 #reset counter
            print('Saving model...')
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter +=1
            print(f'Early stop counter: {early_stopping_counter}')

        if early_stopping_counter > early_stopping_iter:
            print('Early stopping...')
            break  
    
    return

def run_testing(train_loader, valid_loader, test_loader, model, task_type, model_path, out_path, mean, std):
    model.load_state_dict(torch.load(model_path))
    if task_type == 'classification':
        eng = Engine_class(model, optimizer, device=DEVICE)
    else:
        eng = Engine_regression(model, optimizer, mean, std, device=DEVICE)

    print('Begin testing...')
    train_score = eng.validate(train_loader)
    valid_score = eng.validate(valid_loader)
    test_score = eng.validate(test_loader, out_path)
    print('Test completed!')
    return train_score, valid_score, test_score


parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, help="Name of the data")
parser.add_argument("--task_type", type=str, choices=['classification', 'regression'], help="type of the data")
parser.add_argument("--model_path", type=str, default='./model', help="model_path")
parser.add_argument("--prediction_path", type=str, default='./prediction', help="prediction_path")
parser.add_argument("--patience", type=int, default=30, help="patience")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
args = parser.parse_args()

# Print the parsed arguments
print("Parsed arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

params_vertical_gnn['learning_rate'] = args.learning_rate
params = params_vertical_gnn
print(params)

PATIENCE = args.patience

#load dataset 
train_dataset = LoadDataset(root='./data', data_name=args.data_name, split='training')
if args.task_type == 'classification':
    mean, std = None, None
else:
    mean, std = train_dataset.get_scaler()
valid_dataset = LoadDataset(root='./data', data_name=args.data_name, split='valid')
test_dataset = LoadDataset(root='./data', data_name=args.data_name, split='test')

result_pd = pd.DataFrame()
if args.task_type == 'classification':
    result_pd['index'] = ['roc_auc', 'roc_prc', 'accuracy']
else:
    result_pd['index'] = ['r2', 'rmse', 'mae']
start = time.time()
for i in range(5):

    seed = 2024 + i*10
    print(f'------{seed}------')
    seed_everything(seed)

    train_loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)

    model = VerticalGNN(num_features=NUM_FEATURES, num_targets=NUM_TARGET, num_gin_layers=params['num_gin_layers'], num_graph_trans_layers=params['num_graph_trans_layers'], 
                            hidden_size=params['hidden_size'], n_heads=params['n_heads'], dropout=params['dropout'], edge_dim=EDGE_DIM)
    model.to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr = params['learning_rate'])

    run_training(train_loader, valid_loader, test_loader, 
                 model, args.task_type, 
                 os.path.join(args.model_path, f'{args.data_name}_{i+1}.pt'),
                 mean, std)
    train_score, valid_score, test_score = run_testing(train_loader, valid_loader, test_loader, 
                                                       model, args.task_type, 
                                                       os.path.join(args.model_path, f'{args.data_name}_{i+1}.pt'), 
                                                       os.path.join(args.prediction_path, f'{args.data_name}_{i+1}_test_result.csv'),
                                                       mean, std)
    result_pd['train_' + str(i+1)] = train_score
    result_pd['val_' + str(i+1)] = valid_score
    result_pd['test_' + str(i+1)] = test_score
    print(result_pd[['index', 'train_' + str(i + 1), 'val_' + str(i + 1), 'test_' + str(i + 1)]])

elapsed = (time.time() - start)
m, s = divmod(elapsed, 60)
h, m = divmod(m, 60)
print("{} time used:, {:d}:{:d}:{:d}".format(args.data_name, int(h), int(m), int(s)), flush=True)

result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
result_pd.to_csv('./result/Vertical-GNN_' + args.data_name + '_all_result.csv', index=False)
