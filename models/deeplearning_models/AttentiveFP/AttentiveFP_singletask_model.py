import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import pickle as pkl
import os
import sys
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from AttentiveFP import build_dataset
from AttentiveFP.MY_GNN import collate_molgraphs, EarlyStopping, run_a_train_epoch, run_an_eval_epoch_detail, \
    set_random_seed, AttentiveFPPredictor, pos_weight

def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# fix parameters of model
def AttentiveFP_model(times, task_name, data_name, use_scaler=True, scaler=None, classification=False):
    start = time.time()
    args = {}
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'bond'
    args['classification_metric_name'] = ['roc_auc', 'roc_prc', 'accuracy']
    args['regression_metric_name'] = ['r2', 'mae', 'rmse']
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 50
    args['batch_size'] = 128
    
    args['in_feats'] = 40
    args['number_layers'] = 6
    args['num_timesteps'] = 2
    args['graph_feat_size'] = 200
    args['drop_out'] = 0.2
    args['lr'] = 1e-3
    args['weight_decay'] = 1e-5
    args['classification'] = classification
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name
    args['data_name'] = data_name
    args['bin_g_attentivefp_path'] = './data/Attentivefp_graph_data/' + args['data_name'] + '.bin'
    args['group_path'] = './data/Attentivefp_graph_data/' + args['data_name'] + '_group.csv'
    args['times'] = times
    args['use_scaler'] = use_scaler
    args['scaler'] = scaler

    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []
    result_pd = pd.DataFrame()
    if args['classification']:
        args['task_class'] = 'classification'
        result_pd['index'] = args['classification_metric_name']
        args['metric'] = args['classification_metric_name'][0]
        args['mode'] = 'higher'
    else:
        args['task_class'] = 'regression'
        result_pd['index'] = args['regression_metric_name']
        args['metric'] = args['regression_metric_name'][0]
        args['mode'] = 'higher'
    start = time.time()
    for time_id in range(args['times']):
        set_random_seed(2024+time_id*10)
        print('***************************************************************************************************', flush=True)
        print('{}, {}/{} time'.format(args['data_name'], time_id + 1, args['times']), flush=True)
        print('***************************************************************************************************', flush=True)

        train_set, val_set, test_set, mean, std = build_dataset.load_data_for_attentivefp(
            g_path=args['bin_g_attentivefp_path'],
            g_group_path=args['group_path'],
            task_type=args['task_class'],
            use_scaler=args['use_scaler'],
            scaler=args['scaler']
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
        
        pos_weight_np = pos_weight(train_set, 1)
        loss_criterion_c = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np.to(args['device']))
        loss_criterion_r = torch.nn.MSELoss(reduction='none')

        model = AttentiveFPPredictor(node_feat_size=args['in_feats'],
                                     edge_feat_size=10,
                                     num_layers=args['number_layers'],
                                     num_timesteps=args['num_timesteps'],
                                     graph_feat_size=args['graph_feat_size'],
                                     dropout=args['drop_out'])
        print(count_parameters(model))
        optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['data_name']+'_'+str(time_id+1), mode=args['mode'])
        model.to(args['device'])    

        for epoch in range(args['num_epochs']):
            # Train
            _, total_loss = run_a_train_epoch(args, epoch, model, train_loader, loss_criterion_c, loss_criterion_r, optimizer, mean, std)
            # Validation and early stop
            train_score = run_an_eval_epoch_detail(args, model, train_loader, mean, std)[0][0]
            val_score = run_an_eval_epoch_detail(args, model, val_loader, mean, std)[0][0]
            test_score = run_an_eval_epoch_detail(args, model, test_loader, mean, std)[0][0]
            early_stop = stopper.step(val_score, model)
            print('epoch: {:d}/{:d}, {}, lr: {:.6f}, loss: {:.4f}, train: {:.4f}, valid: {:.4f}, best valid score: {:.4f}, test: {:.4f}'.format(
                epoch + 1, args['num_epochs'], args['metric'], args['lr'], total_loss, 
                train_score, val_score, stopper.best_score, test_score), flush=True)
            if early_stop:
                break
        stopper.load_checkpoint(model)
        train_result = run_an_eval_epoch_detail(args, model, train_loader, mean, std)
        val_result = run_an_eval_epoch_detail(args, model, val_loader, mean, std)
        test_result = run_an_eval_epoch_detail(args, model, test_loader, mean, std, out_path=f'./prediction/AttentiveFP_{args["data_name"]}_{time_id+1}_test')
        train_score = train_result[0][0]
        val_score = val_result[0][0]
        test_score = test_result[0][0]
        result_pd['train_' + str(time_id + 1)] = sum(train_result, [])
        result_pd['val_' + str(time_id + 1)] = sum(val_result, [])
        result_pd['test_' + str(time_id + 1)] = sum(test_result, [])
        print(result_pd[['index', 'train_' + str(time_id + 1), 'val_' + str(time_id + 1), 'test_' + str(time_id + 1)]], flush=True)
        print('********************************{}, {}th_time_result*******************************'.format(args['data_name'], time_id + 1), flush=True)
        print("training_result:", round(train_score, 4), flush=True)
        print("val_result:", round(val_score, 4), flush=True)
        print("test_result:", round(test_score, 4), flush=True)
        
        all_times_train_result.append(train_score)
        all_times_val_result.append(val_score)
        all_times_test_result.append(test_score)

        print("************************************{}_times_result************************************".format(time_id + 1), flush=True)
        print('the train result of all tasks ({}): '.format(args['metric']), np.array(all_times_train_result), flush=True)
        print('the average train result of all tasks ({}): {:.4f}'.format(args['metric'], np.array(all_times_train_result).mean()), flush=True)
        print('the train result of all tasks (std): {:.4f}'.format(np.array(all_times_train_result).std()), flush=True)
        print('the train result of all tasks (var): {:.4f}'.format(np.array(all_times_train_result).var()), flush=True)

        print('the val result of all tasks ({}): '.format(args['metric']), np.array(all_times_val_result), flush=True)
        print('the average val result of all tasks ({}): {:.4f}'.format(args['metric'], np.array(all_times_val_result).mean()), flush=True)
        print('the val result of all tasks (std): {:.4f}'.format(np.array(all_times_val_result).std()), flush=True)
        print('the val result of all tasks (var): {:.4f}'.format(np.array(all_times_val_result).var()), flush=True)

        print('the test result of all tasks ({}):'.format(args['metric']), np.array(all_times_test_result), flush=True)
        print('the average test result of all tasks ({}): {:.4f}'.format(args['metric'], np.array(all_times_test_result).mean()), flush=True)
        print('the test result of all tasks (std): {:.4f}'.format(np.array(all_times_test_result).std()), flush=True)
        print('the test result of all tasks (var): {:.4f}'.format(np.array(all_times_test_result).var()), flush=True)
    os.makedirs('./result/hyperparameter/', exist_ok=True)
    with open('./result/hyperparameter/hyperparameter_{}.pkl'.format(args['data_name']), 'wb') as f:
        pkl.dump(args, f, pkl.HIGHEST_PROTOCOL)

    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("{} time used: {:d}:{:d}:{:d}".format(args['data_name'], int(h), int(m), int(s)), flush=True)

    result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
    result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
    result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
    result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
    result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
    result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
    result_pd.to_csv('./result/AttentiveFP_' + args['data_name'] + '_all_result.csv', index=False)
