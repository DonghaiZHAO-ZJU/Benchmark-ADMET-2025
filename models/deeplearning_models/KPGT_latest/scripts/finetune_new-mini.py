import sys
sys.path.append('..')
import os
from src.utils import set_random_seed
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
import numpy as np
import pandas as pd
import random
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset
from src.data.collator import Collator_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.finetune_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict
import time

import warnings
warnings.filterwarnings("ignore")
def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--times", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=100)

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str)
    # parser.add_argument("--datasplit_method", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--use_scaler", action="store_false")
    # parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--metric", nargs="+", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)

    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()
    return args

def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers-2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)
def finetune(args):
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = Collator_tune(config['path_length'])
    train_dataset = MoleculeDataset(root_path=args.data_path, dataset = args.dataset, dataset_type=args.dataset_type, use_scaler=args.use_scaler, split_name=f'{args.split}', split='train')
    val_dataset = MoleculeDataset(root_path=args.data_path, dataset = args.dataset, dataset_type=args.dataset_type, use_scaler=args.use_scaler, split_name=f'{args.split}', split='val')
    test_dataset = MoleculeDataset(root_path=args.data_path, dataset = args.dataset, dataset_type=args.dataset_type, use_scaler=args.use_scaler, split_name=f'{args.split}', split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    # Model Initialization
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=args.dropout,
        feat_drop=args.dropout,
        n_node_types=vocab.vocab_size
    ).to(device)
    # Finetuning Setting
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(f'{args.model_path}').items()})
    model.predictor = get_predictor(d_input_feats=config['d_g_feats']*3, n_tasks=train_dataset.n_tasks, n_layers=2, predictor_drop=args.dropout, device=device, d_hidden_feats=256)
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1e6), flush=True)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.n_epochs*len(train_dataset)//32//10, tot_updates=args.n_epochs*len(train_dataset)//32,lr=args.lr, end_lr=1e-9,power=1)
    
    if args.dataset_type == 'classification':
        loss_fn = BCEWithLogitsLoss(reduction='none', pos_weight=train_dataset._task_pos_weights.to(device))
    else:
        loss_fn = MSELoss(reduction='none')
    if args.dataset_type == 'classification':
        evaluator0 = Evaluator(args.dataset, args.metric[0], train_dataset.n_tasks)
        evaluator1 = Evaluator(args.dataset, args.metric[1], train_dataset.n_tasks)
        evaluator2 = Evaluator(args.dataset, args.metric[2], train_dataset.n_tasks)
    else:
        print('train_dataset.mean:', train_dataset.mean, 'train_dataset.std:', train_dataset.std)
        evaluator0 = Evaluator(args.dataset, args.metric[0], train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
        evaluator1 = Evaluator(args.dataset, args.metric[1], train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
        evaluator2 = Evaluator(args.dataset, args.metric[2], train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    result_tracker = Result_Tracker(args.metric[0])
    summary_writer = None
    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator0, evaluator1, evaluator2, result_tracker, summary_writer, device=device,model_name='LiGhT', label_mean=train_dataset.mean.to(device) if train_dataset.mean is not None else None, label_std=train_dataset.std.to(device) if train_dataset.std is not None else None)
    best_train, best_val, best_test, best_train1, best_val1, best_test1, best_train2, best_val2, best_test2 = trainer.fit(model, train_loader, val_loader, test_loader, model_path=f'../model-mini/model_mini.pth')
    print(f"train: {best_train:.4f}, val: {best_val:.4f}, test: {best_test:.4f}", flush=True)
    model.load_state_dict(torch.load(f'../model-mini/model_mini.pth')['model_state_dict'])
    end_test_result = trainer.eval(model, test_loader, out_path=f'../prediction/KPGT_{args.dataset}_{args.split}_{time_id+1}_test_prediction.csv')
    print('end_test_result:', end_test_result)
    return best_train, best_val, best_test, best_train1, best_val1, best_test1, best_train2, best_val2, best_test2

if __name__ == '__main__':
    args = parse_args()
    print(args, flush=True)
    result_pd = pd.DataFrame()
    result_pd['index'] = args.metric
    start = time.time()
    for time_id in range(args.times):
        print(f'args.seed: {args.seed}')
        set_random_seed(args.seed)
        best_train, best_val, best_test, best_train1, best_val1, best_test1, best_train2, best_val2, best_test2 = finetune(args)
        # best_train, best_val, best_test = 0.0006+time_id, 0.0007+time_id, 0.0008+time_id
        result_pd['train_' + str(time_id + 1)] = [best_train, best_train1, best_train2]
        result_pd['val_' + str(time_id + 1)] = [best_val, best_val1, best_val2]
        result_pd['test_' + str(time_id + 1)] = [best_test, best_test1, best_test2]
        args.seed += 10
    
    result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
    result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
    result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
    result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
    result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
    result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
    os.makedirs('../result/', exist_ok=True)
    result_pd.to_csv('../result/KPGT_' + args.dataset + '_' + args.split + '_all_result.csv', index=False)

    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print(f"Time used on {args.dataset}:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)), flush=True)
    
    

    
    
    
    
    


