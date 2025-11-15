import numpy as np
import build_dataset
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from Model import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, RGCN, pos_weight
from dgl.data.utils import load_graphs
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle as pkl
import os
import time

def train_RGCN(times, task_name, data_name, split_method, scaler=None, classification=False, use_chirality=True):
    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'

    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 50
    args['batch_size'] = 128
    if use_chirality:
        args['in_feats'] = 40
        args['num_rels'] = 64*21
    else:
        args['in_feats'] = 37
        args['num_rels'] = 16*21
    args['rgcn_hidden_feats'] = [64, 64]
    # args['rgcn_hidden_feats'] = [64, 64, 64, 64, 64, 64]
    args['ffn_hidden_feats'] = 64
    args['rgcn_drop_out'] = 0.2
    args['ffn_drop_out'] = 0.2
    args['lr'] = 3
    args['weight_decay'] = 5
    args['classification'] = classification
    args['use_scaler'] = True
    args['scaler'] = scaler
    args['loop'] = True

    # task name (model name)
    args['task_name'] = task_name  # change
    args['data_name'] = data_name  # change
    args['split_method'] = split_method  # change
    args['bin_path'] = os.path.join('./data/graph_data/', f'{data_name}.bin')
    args['group_path'] = os.path.join('./data/graph_data/', f'{data_name}_group.csv')
    args['times'] = times

    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []
    result_pd = pd.DataFrame()
    if args['classification']:
        result_pd['index'] = ['roc_auc', 'roc_prc', 'accuracy', 'sensitivity', 'specificity', 'support', 'f1-score', 'precision', 'recall', 'error rate', 'mcc']
        args['metric_name'] = 'roc_auc'
        args['mode'] = 'higher'
        args['task_type'] = 'classification'
    else:
        result_pd['index'] = ['r2', 'mae', 'rmse']
        args['metric_name'] = 'r2'
        args['mode'] = 'higher'
        args['task_type'] = 'regression'
    start = time.time()
    for time_id in range(args['times']):
        set_random_seed(2024+time_id*10)
        print('***************************************************************************************************', flush=True)
        print('{}, {}/{} time'.format(args['data_name'], time_id + 1, args['times']), flush=True)
        print('***************************************************************************************************', flush=True)

        train_set, val_set, test_set, mean, std = build_dataset.load_data_for_rgcn(
            g_path=args['bin_path'],
            g_group_path=args['group_path'],
            task_type=args['task_type'],
            use_scaler=args['use_scaler'],
            scaler=args['scaler']
        )

        print("Molecule graph is loaded!")
        train_loader = DataLoader(dataset=train_set,
                                batch_size=args['batch_size'],
                                shuffle=True,
                                collate_fn=collate_molgraphs,
                                drop_last=True)

        val_loader = DataLoader(dataset=val_set,
                                batch_size=args['batch_size'],
                                collate_fn=collate_molgraphs)

        test_loader = DataLoader(dataset=test_set,
                                batch_size=args['batch_size'],
                                collate_fn=collate_molgraphs)

        if args['classification']:
            pos_weight_np = pos_weight(train_set)
            loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none',
                                                        pos_weight=pos_weight_np.to(args['device']))
        else:
            loss_criterion = torch.nn.MSELoss(reduction='none')

        model = RGCN(ffn_hidden_feats=args['ffn_hidden_feats'],
                     ffn_dropout=args['ffn_drop_out'],
                     rgcn_node_feats=args['in_feats'], num_rels=args['num_rels'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                     rgcn_drop_out=args['rgcn_drop_out'],
                     classification=args['classification'])
        optimizer = Adam(model.parameters(), lr=10**-args['lr'], weight_decay=10**-args['weight_decay'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['data_name']+'_'+str(time_id+1), mode=args['mode'])
        model.to(args['device'])

        for epoch in range(args['num_epochs']):
            # Train
            _, total_loss = run_a_train_epoch(args, model, train_loader, loss_criterion, optimizer, mean, std)
            # Validation and early stop
            train_score, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion, mean, std, out_path=None)
            val_score, val_loss = run_an_eval_epoch(args, model, val_loader, loss_criterion, mean, std, out_path=None)
            test_score, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion, mean, std, out_path=f'./prediction/RGCN_{data_name}_{time_id+1}_test')
            early_stop = stopper.step(val_score[0], model)
            print('epoch {:d}/{:d}, {}, lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}  train: {:.4f}, valid: {:.4f}, best valid score {:.4f}, '
                  'test: {:.4f}'.format(
                  epoch + 1, args['num_epochs'], args['metric_name'], 10**-args['lr'], total_loss, val_loss, train_score[0], val_score[0],
                  stopper.best_score, test_score[0]), flush=True)
            if early_stop:
                break
        stopper.load_checkpoint(model)

        stop_train_list, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion, mean, std, out_path=None)
        stop_val_list, _ = run_an_eval_epoch(args, model, val_loader, loss_criterion, mean, std, out_path=None)
        stop_test_list, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion, mean, std, out_path=None)
        train_score = stop_train_list[0]
        val_score = stop_val_list[0]
        test_score = stop_test_list[0]
        result_pd['train_' + str(time_id + 1)] = stop_train_list
        result_pd['val_' + str(time_id + 1)] = stop_val_list
        result_pd['test_' + str(time_id + 1)] = stop_test_list
        print(result_pd[['index', 'train_' + str(time_id + 1), 'val_' + str(time_id + 1), 'test_' + str(time_id + 1)]])
        print('********************************{}, {}th_time_result*******************************'.format(args['data_name'], time_id + 1))
        print("training_result:", round(train_score, 4))
        print("val_result:", round(val_score, 4))
        print("test_result:", round(test_score, 4))

        all_times_train_result.append(train_score)
        all_times_val_result.append(val_score)
        all_times_test_result.append(test_score)

        print("************************************{}_times_result************************************".format(time_id + 1))
        print('the train result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_train_result))
        print('the average train result of all tasks ({}): {:.4f}'.format(args['metric_name'], np.array(all_times_train_result).mean()))
        print('the train result of all tasks (std): {:.4f}'.format(np.array(all_times_train_result).std()))
        print('the train result of all tasks (var): {:.4f}'.format(np.array(all_times_train_result).var()))

        print('the val result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_val_result))
        print('the average val result of all tasks ({}): {:.4f}'.format(args['metric_name'], np.array(all_times_val_result).mean()))
        print('the val result of all tasks (std): {:.4f}'.format(np.array(all_times_val_result).std()))
        print('the val result of all tasks (var): {:.4f}'.format(np.array(all_times_val_result).var()))

        print('the test result of all tasks ({}):'.format(args['metric_name']), np.array(all_times_test_result))
        print('the average test result of all tasks ({}): {:.4f}'.format(args['metric_name'], np.array(all_times_test_result).mean()))
        print('the test result of all tasks (std): {:.4f}'.format(np.array(all_times_test_result).std()))
        print('the test result of all tasks (var): {:.4f}'.format(np.array(all_times_test_result).var()))
    os.makedirs('./result/hyperparameter/', exist_ok=True)
    with open('./result/hyperparameter/hyperparameter_{}.pkl'.format(args['data_name']), 'wb') as f:
        pkl.dump(args, f, pkl.HIGHEST_PROTOCOL)

    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("{} time used:, {:d}:{:d}:{:d}".format(args['data_name'], int(h), int(m), int(s)), flush=True)

    result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
    result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
    result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
    result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
    result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
    result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
    result_pd.to_csv('./result/RGCN_' + args['data_name'] + '_all_result.csv', index=False)








































































def val_RGCN(times, task_name, data_name, split_method, classification=False):
    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'

    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 50
    args['batch_size'] = 128
    args['in_feats'] = 40
    args['rgcn_hidden_feats'] = [64, 64]
    args['ffn_hidden_feats'] = 64
    args['rgcn_drop_out'] = 0.2
    args['ffn_drop_out'] = 0.2
    args['lr'] = 3
    args['weight_decay'] = 5
    args['classification'] = classification
    args['loop'] = True

    # task name (model name)
    args['task_name'] = task_name  # change
    args['data_name'] = data_name  # change
    args['split_method'] = split_method  # change
    args['bin_path'] = os.path.join('./data/graph_data/', args['task_name'], args['split_method'], f'{data_name}.bin')
    args['group_path'] = os.path.join('./data/graph_data/', args['task_name'], args['split_method'], f'{data_name}_group.csv')
    args['times'] = times

    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []
    result_pd = pd.DataFrame()
    if args['classification']:
        result_pd['index'] = ['roc_auc', 'roc_prc', 'accuracy', 'sensitivity', 'specificity', 'support', 'f1-score', 'precision', 'recall', 'error rate', 'mcc']
        args['metric_name'] = 'roc_auc'
        args['mode'] = 'higher'
    else:
        result_pd['index'] = ['r2', 'mae', 'rmse']
        args['metric_name'] = 'r2'
        args['mode'] = 'higher'
    start = time.time()
    for time_id in range(args['times']):
        set_random_seed(2024+time_id*10)
        print('***************************************************************************************************', flush=True)
        print('{}, {}/{} time'.format(args['data_name'], time_id + 1, args['times']), flush=True)
        print('***************************************************************************************************', flush=True)

        train_set, val_set, test_set = build_dataset.load_data_for_rgcn(
            g_path=args['bin_path'],
            g_group_path=args['group_path']
        )

        print("Molecule graph is loaded!")
        train_loader = DataLoader(dataset=train_set,
                                batch_size=args['batch_size'],
                                shuffle=True,
                                collate_fn=collate_molgraphs)

        val_loader = DataLoader(dataset=val_set,
                                batch_size=args['batch_size'],
                                collate_fn=collate_molgraphs)

        test_loader = DataLoader(dataset=test_set,
                                batch_size=args['batch_size'],
                                collate_fn=collate_molgraphs)

        if args['classification']:
            pos_weight_np = pos_weight(train_set)
            loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none',
                                                        pos_weight=pos_weight_np.to(args['device']))
        else:
            loss_criterion = torch.nn.MSELoss(reduction='none')

        model = RGCN(ffn_hidden_feats=args['ffn_hidden_feats'],
                     ffn_dropout=args['ffn_drop_out'],
                     rgcn_node_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                     rgcn_drop_out=args['rgcn_drop_out'],
                     classification=args['classification'])
        optimizer = Adam(model.parameters(), lr=10**-args['lr'], weight_decay=10**-args['weight_decay'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['data_name']+'_'+str(time_id+1), mode=args['mode'])
        model.to(args['device'])

        for epoch in range(args['num_epochs']):
            # Train
            _, total_loss = run_a_train_epoch(args, model, train_loader, loss_criterion, optimizer)
            # Validation and early stop
            train_score, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion, out_path=None)
            val_score, val_loss = run_an_eval_epoch(args, model, val_loader, loss_criterion, out_path=None)
            test_score, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion, out_path=None)
            early_stop = stopper.step(val_score[0], model)
            print('epoch {:d}/{:d}, {}, lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}  train: {:.4f}, valid: {:.4f}, best valid score {:.4f}, '
                  'test: {:.4f}'.format(
                  epoch + 1, args['num_epochs'], args['metric_name'], 10**-args['lr'], total_loss, val_loss, train_score[0], val_score[0],
                  stopper.best_score, test_score[0]), flush=True)
            if early_stop:
                break
        stopper.load_checkpoint(model)

        stop_train_list, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion, out_path=None)
        stop_val_list, _ = run_an_eval_epoch(args, model, val_loader, loss_criterion, out_path=None)
        stop_test_list, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion, out_path=None)
        train_score = stop_train_list[0]
        val_score = stop_val_list[0]
        test_score = stop_test_list[0]
        result_pd['train_' + str(time_id + 1)] = stop_train_list
        result_pd['val_' + str(time_id + 1)] = stop_val_list
        result_pd['test_' + str(time_id + 1)] = stop_test_list
        print(result_pd[['index', 'train_' + str(time_id + 1), 'val_' + str(time_id + 1), 'test_' + str(time_id + 1)]])
        print('********************************{}, {}th_time_result*******************************'.format(args['data_name'], time_id + 1))
        print("training_result:", round(train_score, 4))
        print("val_result:", round(val_score, 4))
        print("test_result:", round(test_score, 4))

        all_times_train_result.append(train_score)
        all_times_val_result.append(val_score)
        all_times_test_result.append(test_score)

        print("************************************{}_times_result************************************".format(time_id + 1))
        print('the train result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_train_result))
        print('the average train result of all tasks ({}): {:.4f}'.format(args['metric_name'], np.array(all_times_train_result).mean()))
        print('the train result of all tasks (std): {:.4f}'.format(np.array(all_times_train_result).std()))
        print('the train result of all tasks (var): {:.4f}'.format(np.array(all_times_train_result).var()))

        print('the val result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_val_result))
        print('the average val result of all tasks ({}): {:.4f}'.format(args['metric_name'], np.array(all_times_val_result).mean()))
        print('the val result of all tasks (std): {:.4f}'.format(np.array(all_times_val_result).std()))
        print('the val result of all tasks (var): {:.4f}'.format(np.array(all_times_val_result).var()))

        print('the test result of all tasks ({}):'.format(args['metric_name']), np.array(all_times_test_result))
        print('the average test result of all tasks ({}): {:.4f}'.format(args['metric_name'], np.array(all_times_test_result).mean()))
        print('the test result of all tasks (std): {:.4f}'.format(np.array(all_times_test_result).std()))
        print('the test result of all tasks (var): {:.4f}'.format(np.array(all_times_test_result).var()))
    os.makedirs('./result_for_oral/hyperparameter/', exist_ok=True)
    with open('./result_for_oral/hyperparameter/hyperparameter_{}.pkl'.format(args['data_name']), 'wb') as f:
        pkl.dump(args, f, pkl.HIGHEST_PROTOCOL)

    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("{} time used:, {:d}:{:d}:{:d}".format(args['data_name'], int(h), int(m), int(s)), flush=True)

    result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
    result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
    result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
    result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
    result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
    result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
    result_pd.to_csv('./result_for_oral/RGCN_' + args['data_name'] + '_all_result.csv', index=False)