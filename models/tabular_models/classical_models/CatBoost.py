import os
import time
import argparse

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, CatBoostRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import metrics

def pr_auc_score(y_true, y_pred):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--task_type", type=str, required=True)
parser.add_argument("--fp_name", type=str, required=True)
parser.add_argument("--seed", type=int, default=2024)
args = parser.parse_args()
print("arguments\t", args, flush=True)

start = time.time()
space = {
    'depth': hp.choice('depth', list(range(3, 12))),
    'iterations': hp.quniform('iterations', 100, 1000, 50),
    'lr': hp.choice('lr', [0.1, 0.3, 0.01, 0.03, 0.05, 0.001, 0.003, 0.005]),
    'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 10, 1),
    'border_count': hp.quniform('border_count', 32, 255, 16),
    'bagging_temperature': hp.choice('bagging_temperature', [0, 0.2, 0.4, 0.6, 0.8, 1.0]), 
    'grow_policy': hp.choice('grow_policy', ['SymmetricTree', 'Depthwise'])
}

data = pd.read_csv(f'data/processed_data/{args.fp_name}/{args.data_name}_{args.fp_name}.csv')
features = data.columns.difference(['smiles', args.task_name, 'group'])

train_set = data[data['group']=='training']
valid_set = data[data['group']=='valid']
test_set = data[data['group']=='test']

train_x, train_y = train_set[features], train_set[args.task_name]
valid_x, valid_y = valid_set[features], valid_set[args.task_name]
test_x, test_y = test_set[features], test_set[args.task_name]
test_smiles = test_set['smiles']

def hyperopt_my_catb_classification(parameter):
    model = CatBoostClassifier(
        learning_rate=parameter['lr'],
        depth=parameter['depth'],
        iterations=parameter['iterations'],
        l2_leaf_reg=parameter['l2_leaf_reg'],
        border_count=parameter['border_count'],
        bagging_temperature=parameter['bagging_temperature'],
        grow_policy=parameter['grow_policy'],
        random_seed=2044,
        verbose=0,
        thread_count=4,
        # auto_class_weights='Balanced',
        # task_type="GPU",
        # gpu_ram_part=0.80
    )
    model.fit(train_x, train_y)

    valid_prediction = model.predict_proba(valid_x)[:, 1]
    auc = metrics.roc_auc_score(valid_y, valid_prediction)
    # auc = pr_auc_score(valid_y, valid_prediction)
    return {'loss':-auc, 'status':STATUS_OK, 'model':model}
    # valid_prediction = model.predict(valid_x)
    # acc_score = metrics.accuracy_score(valid_y, valid_prediction)
    # return {'loss':-acc_score, 'status':STATUS_OK, 'model':model}

def hyperopt_my_catb_regression(parameter):
    model = CatBoostRegressor(
        learning_rate=parameter['lr'],
        depth=parameter['depth'],
        iterations=parameter['iterations'],
        l2_leaf_reg=parameter['l2_leaf_reg'],
        border_count=parameter['border_count'],
        bagging_temperature=parameter['bagging_temperature'],
        grow_policy=parameter['grow_policy'],
        random_seed=2044,
        verbose=0,
        thread_count=4,
        # task_type="GPU",
        # gpu_ram_part=0.80
    )
    model.fit(train_x, train_y)

    valid_prediction = model.predict(valid_x)
    r2 = metrics.r2_score(valid_y, valid_prediction)
    return {'loss':-r2, 'status':STATUS_OK, 'model':model}

trials = Trials()
if args.task_type=='classification':
    best = fmin(hyperopt_my_catb_classification, space, algo=tpe.suggest, trials=trials, max_evals=30)
else:
    best = fmin(hyperopt_my_catb_regression, space, algo=tpe.suggest, trials=trials, max_evals=30)
print(best)

args.depth = list(range(3, 12))[best['depth']]
args.lr = [0.1, 0.3, 0.01, 0.03, 0.05, 0.001, 0.003, 0.005][best['lr']]
args.iterations = int(best['iterations'])
args.l2_leaf_reg = int(best['l2_leaf_reg'])
args.border_count = int(best['border_count'])
args.bagging_temperature = [0, 0.2, 0.4, 0.6, 0.8, 1.0][best['bagging_temperature']]
args.grow_policy = ['SymmetricTree', 'Depthwise'][best['grow_policy']]

result_pd = pd.DataFrame()
if args.task_type=='classification':
    result_pd['index'] = ['roc_auc','roc_prc','accuracy']
    for i in range(5):
        print(f'------time: {i+1}------')
        model = CatBoostClassifier(
            learning_rate=args.lr,
            depth=args.depth,
            iterations=int(args.iterations),
            l2_leaf_reg=int(args.l2_leaf_reg), 
            border_count=int(args.border_count), 
            bagging_temperature=args.bagging_temperature, 
            grow_policy=args.grow_policy, 
            random_seed=2024+i*10,
            verbose=0,
            thread_count=4,
            # auto_class_weights='Balanced',
            # task_type="GPU",
            # gpu_ram_part=0.80
        )
        model.fit(train_x, train_y)
        train_prediction, valid_prediction, test_prediction = [model.predict_proba(x)[:, 1] for x in (train_x, valid_x, test_x)]
        prediction = pd.DataFrame({'smiles':test_smiles, 'pred':test_prediction, 'label':test_y})
        os.makedirs(f'./CatBoost_prediction/{args.fp_name}/', exist_ok=True)
        prediction.to_csv(f'./CatBoost_prediction/{args.fp_name}/CatBoost_{args.fp_name}_{args.data_name}_{i+1}_test_prediction.csv', index=False)
        roc_auc_score = metrics.roc_auc_score(train_y, train_prediction), metrics.roc_auc_score(valid_y, valid_prediction), metrics.roc_auc_score(test_y, test_prediction)
        roc_prc_score = pr_auc_score(train_y, train_prediction), pr_auc_score(valid_y, valid_prediction), pr_auc_score(test_y, test_prediction)    
        acc_score = metrics.accuracy_score(train_y, list(map(round, train_prediction))), metrics.accuracy_score(valid_y, list(map(round, valid_prediction))), metrics.accuracy_score(test_y, list(map(round, test_prediction)))
        result_pd['train_' + str(i+1)] = [roc_auc_score[0], roc_prc_score[0], acc_score[0]]
        result_pd['val_' + str(i+1)] = [roc_auc_score[1], roc_prc_score[1], acc_score[1]]
        result_pd['test_' + str(i+1)] = [roc_auc_score[2], roc_prc_score[2], acc_score[2]]
        print(result_pd) 
else:
    result_pd['index'] = ['r2','rmse','mae']
    for i in range(5):
        print(f'------time: {i+1}------')
        model = CatBoostRegressor(
            learning_rate=args.lr,
            depth=args.depth,
            iterations=int(args.iterations),
            l2_leaf_reg=int(args.l2_leaf_reg), 
            border_count=int(args.border_count), 
            bagging_temperature=args.bagging_temperature, 
            grow_policy=args.grow_policy, 
            random_seed=2024+i*10,
            verbose=0,
            thread_count=4,
            # task_type="GPU",
            # gpu_ram_part=0.80
        )
        model.fit(train_x, train_y)
        train_prediction, valid_prediction, test_prediction = [model.predict(x) for x in (train_x, valid_x, test_x)]
        prediction = pd.DataFrame({'smiles':test_smiles, 'pred':test_prediction, 'label':test_y})
        os.makedirs(f'./CatBoost_prediction/{args.fp_name}/', exist_ok=True)
        prediction.to_csv(f'./CatBoost_prediction/{args.fp_name}/CatBoost_{args.fp_name}_{args.data_name}_{i+1}_test_prediction.csv', index=False)
        r2_score = metrics.r2_score(train_y, train_prediction), metrics.r2_score(valid_y, valid_prediction), metrics.r2_score(test_y, test_prediction)
        rmse_score = np.sqrt(metrics.mean_squared_error(train_y, train_prediction)), np.sqrt(metrics.mean_squared_error(valid_y, valid_prediction)), np.sqrt(metrics.mean_squared_error(test_y, test_prediction))
        mae_score = metrics.mean_absolute_error(train_y, train_prediction), metrics.mean_absolute_error(valid_y, valid_prediction), metrics.mean_absolute_error(test_y, test_prediction)
        result_pd['train_' + str(i+1)] = [r2_score[0], rmse_score[0], mae_score[0]]
        result_pd['val_' + str(i+1)] = [r2_score[1], rmse_score[1], mae_score[1]]
        result_pd['test_' + str(i+1)] = [r2_score[2], rmse_score[2], mae_score[2]]
        print(result_pd)              

elapsed = (time.time() - start)
m, s = divmod(elapsed, 60)
h, m = divmod(m, 60)
print("{} {} time used:, {:d}:{:d}:{:d}".format(args.data_name, args.fp_name, int(h), int(m), int(s)), flush=True)

result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
os.makedirs(f'./CatBoost_result/{args.fp_name}/', exist_ok=True)
result_pd.to_csv(f'./CatBoost_result/{args.fp_name}/CatBoost_{args.fp_name}_{args.data_name}_all_result.csv', index=False)  