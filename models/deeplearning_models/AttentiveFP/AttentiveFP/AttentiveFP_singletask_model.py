import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import os
import sys
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from AttentiveFP import build_dataset
from AttentiveFP.MY_GNN import collate_molgraphs, EarlyStopping, run_a_train_epoch, run_an_eval_epoch_detail, \
    set_random_seed, AttentiveFPPredictor, pos_weight


# fix parameters of model
def AttentiveFP_model(times, task_name, data_name,
                      number_layers=2,
                      num_timesteps=2,
                      graph_feat_size=200,
                      lr=1e-3,
                      weight_decay=1e-5,
                      dropout=0.2):
    start = time.time()
    args = {}
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'bond'
    args['classification_metric_name'] = ['accuracy', 'roc_auc', 'roc_prc']
    args['regression_metric_name'] = ['rmse', 'r2', 'mae']
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 50
    args['batch_size'] = 128
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['number_layers'] = number_layers
    args['num_timesteps'] = num_timesteps
    args['graph_feat_size'] = graph_feat_size
    args['drop_out'] = dropout
    args['lr'] = lr
    args['weight_decay'] = weight_decay
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name
    args['data_name'] = data_name
    args['times'] = times

    # selected task, generate select task index, task class, and classification_num
    args['select_task_list'] = [args['task_name']]  # change
    args['select_task_index'] = []
    args['classification_num'] = 0
    args['regression_num'] = 0
    args['all_task_list'] = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability",
                            "Caco2","HalfLife","VDss"]  # change
    # generate select task index
    for index, task in enumerate(args['all_task_list']):
        if task in args['select_task_list']:
            args['select_task_index'].append(index)
    # generate classification_num
    for task in args['select_task_list']:
        if task in ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability"]:
            args['classification_num'] = args['classification_num'] + 1
        if task in ["Caco2","HalfLife","VDss"]:
            args['regression_num'] = args['regression_num'] + 1
    # generate classification_num
    if args['classification_num'] != 0 and args['regression_num'] != 0:
        args['task_class'] = 'classification_regression'
    if args['classification_num'] != 0 and args['regression_num'] == 0:
        args['task_class'] = 'classification'
    if args['classification_num'] == 0 and args['regression_num'] != 0:
        args['task_class'] = 'regression'
    print('Classification task:{}, Regression Task:{}'.format(args['classification_num'], args['regression_num']))
    args['bin_g_attentivefp_path'] = '../data/Attentivefp_graph_data/' + args['data_name'] + '.bin'
    args['group_path'] = '../data/Attentivefp_graph_data/' + args['data_name'] + '_group.csv'

    result_pd = pd.DataFrame(columns=args['select_task_list']+['group'] + args['select_task_list']+['group']
                             + args['select_task_list']+['group'])
    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []

    for time_id in range(args['times']):
        set_random_seed(2020 + time_id)
        print('***************************************************************************************************')
        print('{}, {}/{} time'.format(args['task_name'], time_id + 1, args['times']))
        print('***************************************************************************************************')
        # random split load dataset
        train_set, val_set, test_set, task_number = build_dataset.load_graph_from_csv_bin_for_splited(
            bin_g_attentivefp_path=args['bin_g_attentivefp_path'],
            group_path=args['group_path'],
            select_task_index=args['select_task_index']
        )
        print("Molecule graph is loaded!", flush=True)
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
        
        pos_weight_np = pos_weight(train_set, classification_num=args['classification_num'])
        loss_criterion_c = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np.to(args['device']))
        loss_criterion_r = torch.nn.MSELoss(reduction='none')

        model = AttentiveFPPredictor(n_tasks=task_number,
                                     node_feat_size=args['in_feats'],
                                     edge_feat_size=10,
                                     num_layers=args['number_layers'],
                                     num_timesteps=args['num_timesteps'],
                                     graph_feat_size=args['graph_feat_size'],
                                     dropout=args['drop_out'])

        filename = '../model/AttentiveFP/{}_early_stop.pth'.format(args['task_name'] + '_' + str(time_id + 1))
        stopper = EarlyStopping(patience=args['patience'], mode=args['mode'], filename=filename)
        model.to(args['device'])
        lr = args['lr']
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])

        for epoch in range(args['num_epochs']):
            # Train
            # lr = args['lr']
            _, total_loss = run_a_train_epoch(args, epoch, model, train_loader, loss_criterion_c, loss_criterion_r, optimizer)
            # Validation and early stop
            validation_result = run_an_eval_epoch_detail(args, model, val_loader)[1]
            val_score = np.mean(validation_result)
            early_stop = stopper.step(val_score, model)
            print('epoch {:d}/{:d}, validation {:.4f}, best validation {:.4f}'.format(
                epoch + 1, args['num_epochs'],
                val_score, stopper.best_score), flush=True)
            if early_stop:
                break
        stopper.load_checkpoint(model)
        train_score = run_an_eval_epoch_detail(args, model, train_loader)
        val_score = run_an_eval_epoch_detail(args, model, val_loader)
        test_score = run_an_eval_epoch_detail(args, model, test_loader)
        all_times_train_result.append(train_score)
        all_times_val_result.append(val_score)
        all_times_test_result.append(test_score)
        # deal result
        result = train_score[1] + ['training'] + val_score[1] + ['valid'] + test_score[1] + ['test']
        result_pd.loc[time_id] = result
        print('********************************{}, {}_times_result*******************************'.format(args['task_name'], time_id+1), flush=True)
        print("training_result:", train_score, flush=True)
        print("val_result:", val_score, flush=True)
        print("test_result:", test_score, flush=True)
    os.makedirs('../result/{}_singletask/'.format(data_name), exist_ok=True)
    result_pd.to_csv('../result/{}_singletask/'.format(data_name) + args['task_name'] + '_result.csv', index=False)

    all_times_train_result = np.array(all_times_train_result)
    all_times_val_result = np.array(all_times_val_result)
    all_times_test_result = np.array(all_times_test_result)
    train_average_values = np.mean(all_times_train_result, axis=0)
    train_average_values = np.round(train_average_values, decimals=4)
    train_variance_values = np.std(all_times_train_result, axis=0)
    train_variance_values = np.round(train_variance_values, decimals=4)
    val_average_values = np.mean(all_times_val_result, axis=0)
    val_average_values = np.round(val_average_values, decimals=4)
    val_variance_values = np.std(all_times_val_result, axis=0)
    val_variance_values = np.round(val_variance_values, decimals=4)
    test_average_values = np.mean(all_times_test_result, axis=0)
    test_average_values = np.round(test_average_values, decimals=4)
    test_variance_values = np.std(all_times_test_result, axis=0)
    test_variance_values = np.round(test_variance_values, decimals=4)
    result_pd_val_average = pd.DataFrame(val_average_values, columns=args['select_task_list'])
    result_pd_val_var = pd.DataFrame(val_variance_values, columns=args['select_task_list'])
    result_pd_test_average = pd.DataFrame(test_average_values, columns=args['select_task_list'])
    result_pd_test_var = pd.DataFrame(test_variance_values, columns=args['select_task_list'])
    result_pd_val_average.to_csv('../result/{}_singletask/'.format(data_name)+args['task_name']+'_result_val_average.csv', index=None)
    result_pd_val_var.to_csv('../result/{}_singletask/'.format(data_name)+args['task_name']+'_result_val_var.csv', index=None)
    result_pd_test_average.to_csv('../result/{}_singletask/'.format(data_name)+args['task_name']+'_result_test_average.csv', index=None)
    result_pd_test_var.to_csv('../result/{}_singletask/'.format(data_name)+args['task_name']+'_result_test_var.csv', index=None)
    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("Time used:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)), flush=True)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, help="task name")
    parser.add_argument("--data_name", type=str, help="data name")
    args = parser.parse_args()
    AttentiveFP_model(times=5, task_name=args.task_name, data_name=args.data_name)
