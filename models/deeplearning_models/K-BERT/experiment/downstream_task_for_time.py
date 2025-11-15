from experiment import build_data
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from experiment.my_nn import collate_data, EarlyStopping, run_a_train_global_epoch, run_an_eval_global_epoch,\
    set_random_seed, K_BERT_WCL, pos_weight
import os
import numpy as np
import pandas as pd
import argparse
import time
import pickle as pkl


def train_K_BERT(task_name, data_name, split_method, scaler=None, classification=False, savecheckpoint=True):
    args = {}
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    args['batch_size'] = 128
    args['num_epochs'] = 500
    args['d_model'] = 768
    args['n_layers'] = 6
    args['vocab_size'] = 47
    args['maxlen'] = 201
    args['d_k'] = 64
    args['d_v'] = 64
    args['d_ff'] = 768 * 4
    args['n_heads'] = 12
    args['global_labels_dim'] = 1
    args['atom_labels_dim'] = 15
    if classification:
        args['lr'] = 3e-5
    else:
        args['lr'] = 3e-4
    args['pretrain_layer'] = 5
    args['patience'] = 20
    args['classification'] = classification
    args['pretrain_model'] = 'pretrain_k_bert_wcl_epoch_7.pth'
    # args['pretrain_model'] = 'pretrain_k_bert_epoch_7.pth'


    args['task_name'] = task_name  # change
    args['data_name'] = data_name  # change
    args['split_method'] = split_method  # change
    args['data_path'] = f'./data/token_data/{data_name}.npy'
    args['savecheckpoint'] = savecheckpoint
    args['times'] = 5

    args['scaler'] = scaler
    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []
    result_pd = pd.DataFrame()
    if classification:
        result_pd['index'] = ['roc_auc', 'roc_prc', 'accuracy', 'sensitivity', 'specificity', 'f1-score', 'precision', 'recall', 
                              'error rate', 'mcc']
        args['metric_name'] = 'roc_auc'
        args['mode'] = 'higher'
        task_type = 'classification'
    else:
        result_pd['index'] = ['r2', 'rmse', 'mae']
        args['metric_name'] = 'r2'
        args['mode'] = 'higher'
        task_type = 'regression' 
    start = time.time()

    training_times = []
    testing_times = []

    for time_id in range(args['times']):
        set_random_seed(2024+time_id*10)
        print('***************************************************************************************************', flush=True)
        print('{}, {}/{} time'.format(args['data_name'], time_id + 1, args['times']), flush=True)
        print('***************************************************************************************************', flush=True)

        train_set, val_set, test_set, _ = build_data.load_data_for_splited(
            data_name=args['data_name'],
            data_path=args['data_path'], 
            task_type=task_type, 
            scaler=args['scaler']
        )

        print("Molecule graph is loaded!", flush=True)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args['batch_size'],
                                  shuffle=True,
                                  collate_fn=collate_data)

        val_loader = DataLoader(dataset=val_set,
                                batch_size=args['batch_size'],
                                collate_fn=collate_data)

        test_loader = DataLoader(dataset=test_set,
                                 batch_size=args['batch_size'],
                                 collate_fn=collate_data)

        if task_type == 'classification':
            pos_weight_task = pos_weight(train_set)
            loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none', 
                                                        pos_weight=pos_weight_task.to(args['device']))
        else:
            loss_criterion = torch.nn.MSELoss(reduction='none')

        model = K_BERT_WCL(d_model=args['d_model'], n_layers=args['n_layers'], vocab_size=args['vocab_size'],
                            maxlen=args['maxlen'], d_k=args['d_k'], d_v=args['d_v'], n_heads=args['n_heads'], d_ff=args['d_ff'],
                            global_label_dim=args['global_labels_dim'], atom_label_dim=args['atom_labels_dim'])
        
        stopper = EarlyStopping(patience=args['patience'], pretrained_model=args['pretrain_model'],
                                pretrain_layer=args['pretrain_layer'],
                                task_name=args['task_name'], mode=args['mode'], savecheckpoint=args['savecheckpoint'])
        stopper.load_pretrained_model(model)
        print('parameters num: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        optimizer = Adam(model.parameters(), lr=args['lr'])
        model.to(args['device'])

        # 记录训练开始时间
        train_start = time.time()

        for epoch in range(args['num_epochs']):
            _ = run_a_train_global_epoch(args, epoch, model, train_loader, loss_criterion, optimizer, task_type)
            # Validation and early stop
            train_score = run_an_eval_global_epoch(args, model, train_loader, task_type)[0]
            val_score = run_an_eval_global_epoch(args, model, val_loader, task_type)[0]
            test_score = run_an_eval_global_epoch(args, model, test_loader, task_type)[0]
            if epoch < 5:
                if task_type == 'classification':
                    early_stop = stopper.step(0, model)
                else:
                    early_stop = stopper.step(float('-inf'), model)
            else:
                early_stop = stopper.step(val_score, model)
            print('epoch {:d}/{:d}, {}, lr: {:.6f},  train: {:.4f}, valid: {:.4f}, best valid {:.4f}, '
                  'test: {:.4f}'.format(
                  epoch + 1, args['num_epochs'], args['metric_name'], optimizer.param_groups[0]['lr'], train_score, val_score,
                  stopper.best_score, test_score), flush=True)
            if early_stop:
                break
    
        # 记录训练结束时间
        train_end = time.time()
        training_time = train_end - train_start
        print(f"Training time: {training_time:.2f} seconds")

        stopper.load_checkpoint(model)

        stop_train_list = run_an_eval_global_epoch(args, model, train_loader, task_type)
        stop_val_list = run_an_eval_global_epoch(args, model, val_loader, task_type)

        # 记录测试开始时间
        test_start = time.time()

        stop_test_list = run_an_eval_global_epoch(args, model, test_loader, task_type)
        
        # 记录测试结束时间
        test_end = time.time()
        testing_time = test_end - test_start
        print(f"Testing time: {testing_time:.2f} seconds")  

        # 保存当前运行的耗时
        training_times.append(training_time)
        testing_times.append(testing_time)
      
        train_score = stop_train_list[0]
        val_score = stop_val_list[0]
        test_score = stop_test_list[0]
        result_pd['train_' + str(time_id + 1)] = stop_train_list
        result_pd['val_' + str(time_id + 1)] = stop_val_list
        result_pd['test_' + str(time_id + 1)] = stop_test_list
        print(result_pd[['index', 'train_' + str(time_id + 1), 'val_' + str(time_id + 1), 'test_' + str(time_id + 1)]])
        print('********************************{}, {}th_time_result*******************************'.format(args['data_name'], time_id + 1))
        print("training_result:", round(train_score, 4), flush=True)
        print("val_result:", round(val_score, 4), flush=True)
        print("test_result:", round(test_score, 4), flush=True)

        all_times_train_result.append(train_score)
        all_times_val_result.append(val_score)
        all_times_test_result.append(test_score)
    
        print("************************************{}_times_result************************************".format(time_id + 1), flush=True)
        print('the train result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_train_result), flush=True)
        print('the average train result of all tasks ({}): {:.4f}'.format(args['metric_name'], np.array(all_times_train_result).mean()), flush=True)
        print('the train result of all tasks (std): {:.4f}'.format(np.array(all_times_train_result).std()), flush=True)
        print('the train result of all tasks (var): {:.4f}'.format(np.array(all_times_train_result).var()), flush=True)

        print('the val result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_val_result), flush=True)
        print('the average val result of all tasks ({}): {:.4f}'.format(args['metric_name'], np.array(all_times_val_result).mean()), flush=True)
        print('the val result of all tasks (std): {:.4f}'.format(np.array(all_times_val_result).std()), flush=True)
        print('the val result of all tasks (var): {:.4f}'.format(np.array(all_times_val_result).var()), flush=True)

        print('the test result of all tasks ({}):'.format(args['metric_name']), np.array(all_times_test_result), flush=True)
        print('the average test result of all tasks ({}): {:.4f}'.format(args['metric_name'], np.array(all_times_test_result).mean()), flush=True)
        print('the test result of all tasks (std): {:.4f}'.format(np.array(all_times_test_result).std()), flush=True)
        print('the test result of all tasks (var): {:.4f}'.format(np.array(all_times_test_result).var()), flush=True)
    os.makedirs('./result/hyperparameter/', exist_ok=True)
    with open('./result/hyperparameter/hyperparameter_{}.pkl'.format(args['data_name']), 'wb') as f:
        pkl.dump(args, f, pkl.HIGHEST_PROTOCOL)

    # 创建 DataFrame
    data = {
        "Model": ['K_BERT'],
        **{f"train_{i+1}": [training_times[i]] for i in range(5)},
        **{f"test_{i+1}": [testing_times[i]] for i in range(5)}
    }

    df = pd.DataFrame(data)

    # 显示 DataFrame
    print(df)

    # 保存为 CSV
    df.to_csv("time/K_BERT.csv")

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
    # result_pd.to_csv('./result/K_BERT_' + args['data_name'] + '_all_result.csv', index=False)



















