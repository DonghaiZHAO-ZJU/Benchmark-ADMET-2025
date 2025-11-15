import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from os.path import join, exists, basename
import argparse
import numpy as np
import pandas as pd

import paddle
import paddle.nn as nn
import pgl

from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.datasets.inmemory_dataset import InMemoryDataset

from src.model import DownstreamModel
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from src.utils import calc_rocauc_score, calc_acc_score, calc_rocprc_score, exempt_parameters
from src.self_data import load_my_dataset, GroupSplitter

import random
import time


def set_random_seed(seed=10):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt):
    """
    Define the train function 
    Args:
        args,model,train_dataset,collate_fn,criterion,encoder_opt,head_opt;
    Returns:
        the average of the list loss
    """
    data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=collate_fn)
    list_loss = []
    model.train()
    for atom_bond_graphs, bond_angle_graphs, valids, _, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs)
        loss = criterion(preds, labels)
        loss = paddle.sum(loss * valids) / paddle.sum(valids)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
        list_loss.append(loss.item())
    return np.mean(list_loss)


def evaluate(args, model, test_dataset, collate_fn, out_path=None):
    """
    Define the evaluate function
    In the dataset, a proportion of labels are blank. So we use a `valid` tensor 
    to help eliminate these blank labels in both training and evaluation phase.
    """
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)
    total_smiles = []
    total_pred = []
    total_label = []
    total_valid = []
    result_pd = pd.DataFrame()
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, valids, smiles, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs)
        total_smiles.append(np.array(smiles))
        total_pred.append(preds.numpy())
        total_valid.append(valids.numpy())
        total_label.append(labels.numpy())
    total_smiles = np.concatenate(total_smiles, 0)
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    total_valid = np.concatenate(total_valid, 0)
    if out_path:
        result_pd['smiles'] = total_smiles
        result_pd['pred'] = total_pred
        result_pd['label'] = total_label
        result_pd.to_csv(out_path, index=False)
    return calc_rocauc_score(total_label, total_pred, total_valid), \
           calc_rocprc_score(total_label, total_pred, total_valid), \
           calc_acc_score(total_label, total_pred, total_valid),

def get_pos_neg_ratio(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])
    return np.mean(labels == 1), np.mean(labels == -1)

def main(args, seed=2024):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    model_config:
        a json file  with the hyperparameters,such as dropout rate ,learning rate,num tasks and so on;
    num_tasks:
        it means the number of task that each dataset contains, it's related to the dataset;
    """
    ### config for the body
    print(f'current seed: {seed}')
    set_random_seed(seed)

    compound_encoder_config = load_json_config(args.compound_encoder_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    ### config for the downstream task
    task_type = 'class'

    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate
    model_config['task_type'] = task_type
    model_config['num_tasks'] = 1 # for single-task

    ### build model
    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)
    criterion = nn.BCELoss(reduction='none')
    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)
    encoder_opt = paddle.optimizer.Adam(args.encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(args.head_lr, parameters=head_params)
    print('Total param: ', sum(p.numel() for p in model.parameters() if not p.stop_gradient))
    print('Total param num: %s' % (len(model.parameters())))
    print('Encoder param num: %s' % (len(encoder_params)))
    print('Head param num: %s' % (len(head_params)))
    for i, param in enumerate(model.named_parameters()):
        print(i, param[0], param[1].name)

    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)

    ### load data    
    if args.task == 'data':
        if not os.path.isfile(os.path.join(args.processed_data_path, args.task_name, 'part-000000.npz')):
            print('Preprocessing data...')
            dataset = load_my_dataset(os.path.join(args.data_path, f'{args.task_name}.csv'), task_type=task_type)
            transform_fn = DownstreamTransformFn()
            dataset.transform(transform_fn, num_workers=args.num_workers)
            dataset.save_data(os.path.join(args.processed_data_path, args.task_name))
            print('Saved all data!')    
        else:
            print('Data existed!')
        return
    else:
        try:
            print('Read preprocessing data...')
            dataset = InMemoryDataset(npz_data_path=os.path.join(args.processed_data_path, args.task_name))
        except:
            print('Processing data...')
            dataset = load_my_dataset(os.path.join(args.data_path, f'{args.task_name}.csv'), task_type=task_type)
            transform_fn = DownstreamTransformFn()
            dataset.transform(transform_fn, num_workers=args.num_workers)
            dataset.save_data(os.path.join(args.processed_data_path, args.task_name))
            print('Saved all data!')

    group_dataframe = pd.read_csv(os.path.join(args.group_data_path, f'{args.data_name}.csv'))
    splitter = GroupSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.split(
            dataset, group_dataframe)

    print("Train/Valid/Test num: %s/%s/%s" % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))
    print('Train pos/neg ratio %s/%s' % get_pos_neg_ratio(train_dataset))
    print('Valid pos/neg ratio %s/%s' % get_pos_neg_ratio(valid_dataset))
    print('Test pos/neg ratio %s/%s' % get_pos_neg_ratio(test_dataset))


    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type='class')
    best_epoch = 0
    counter = 0
    best_perfor = 0

    # 记录训练开始时间
    train_start = time.time()
    for epoch_id in range(args.max_epoch):
        train_loss = train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt)
        train_metric = evaluate(args, model, train_dataset, collate_fn)[0]
        val_metric = evaluate(args, model, valid_dataset, collate_fn)[0]
        test_metric = evaluate(args, model, test_dataset, collate_fn)[0]
        print(f'epoch: {epoch_id+1}/{args.max_epoch}, train_loss: {train_loss}, train roc-auc: {train_metric}, valid roc-auc: {val_metric}, test roc-auc: {test_metric}')
        if val_metric > best_perfor:
            best_perfor = val_metric
            best_epoch = epoch_id+1
            paddle.save(compound_encoder.state_dict(), os.path.join(args.model_dir, f'{args.data_name}_{int((seed-2024)/10+1)}', 'compound_encoder.pdparams'))
            paddle.save(model.state_dict(), os.path.join(args.model_dir, f'{args.data_name}_{int((seed-2024)/10+1)}', 'model.pdparams'))
            counter = 0
        else:
            counter += 1
            print(f'out of counter-{counter}')
        
        if (epoch_id+1)-best_epoch>=20:
            print('out of patience-20!')
            break

    # 记录训练结束时间
    train_end = time.time()
    training_time = train_end - train_start
    print(f"Training time: {training_time:.2f} seconds")

    compound_encoder.set_state_dict(paddle.load(os.path.join(args.model_dir, f'{args.data_name}_{int((seed-2024)/10+1)}', 'compound_encoder.pdparams')))
    model.set_state_dict(paddle.load(os.path.join(args.model_dir, f'{args.data_name}_{int((seed-2024)/10+1)}', 'model.pdparams')))
    train_result = evaluate(args, model, train_dataset, collate_fn)
    valid_result = evaluate(args, model, valid_dataset, collate_fn)

    # 记录测试开始时间
    test_start = time.time()

    test_result = evaluate(args, model, test_dataset, collate_fn)

    # 记录测试结束时间
    test_end = time.time()
    testing_time = test_end - test_start
    print(f"Testing time: {testing_time:.2f} seconds")

    return train_result, valid_result, test_result, training_time, testing_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--processed_data_path", type=str, default=None)
    parser.add_argument("--group_data_path", type=str, default=None)

    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--output_dir", type=str, default='./prediction')
    parser.add_argument("--encoder_lr", type=float, default=0.001)
    parser.add_argument("--head_lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--exp_id", type=int, help='used for identification only')
    args = parser.parse_args()

    if args.task == 'data':
        main(args)
    else:
        result_pd = pd.DataFrame()
        result_pd['index'] = ['roc_auc', 'roc_prc', 'accuracy']
        start = time.time()

        training_times = []
        testing_times = []

        for i in range(5):
            seed = 2024+10*i
            train_result, valid_result, test_result, training_time, testing_time = main(args, seed)
            
            # 保存当前运行的耗时
            training_times.append(training_time)
            testing_times.append(testing_time)

            result_pd['train_' + str(i + 1)] = train_result
            result_pd['val_' + str(i + 1)] = valid_result
            result_pd['test_' + str(i + 1)] = test_result

        result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
        result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
        result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
        result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
        result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
        result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
        os.makedirs('./result/', exist_ok=True)
        # result_pd.to_csv('./result/GEM_' + args.data_name + '_all_result.csv', index=False)

        data = {
            "Model": ['GEM'],
            **{f"train_{i+1}": [training_times[i]] for i in range(5)},
            **{f"test_{i+1}": [testing_times[i]] for i in range(5)}
        }

        df = pd.DataFrame(data)

        # 显示 DataFrame
        print(df)

        # 保存为 CSV
        df.to_csv("time/GEM.csv")

        elapsed = (time.time() - start)
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        print(f"Time used on {args.data_name}:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)), flush=True)