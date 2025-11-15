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

def extract_features(args, seed=2024):
    print(f'current seed: {seed}')
    set_random_seed(seed)

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

    collate_fn = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'], 
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type='class')

    data_gen = dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn
    )

    compound_encoder.eval()
    fps_list = []
    for atom_bond_graphs, bond_angle_graphs, valids, smiles, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        _, _, graph_repr = compound_encoder(atom_bond_graphs, bond_angle_graphs)
        fps_list.extend(graph_repr.detach().cpu().numpy().tolist())
    np.savez_compressed(f"features/gem_features_{args.task_name}.npz", fps=np.array(fps_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--task_name", type=str)
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

    extract_features(args)