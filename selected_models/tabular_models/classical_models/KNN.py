import os
import time
import argparse

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
space = {'n_neighbors':hp.choice('n_neighbors', [i for i in range(1, 11)]),
         'weights': hp.choice('weights', ['uniform', 'distance']),
         'p': hp.choice('p', [1,2,3,4,5]),
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

def hyperopt_my_xgb_classification(parameter):
    print(f"Trying parameters: {parameter}")
    model = KNeighborsClassifier(n_neighbors=parameter['n_neighbors'],
                                 weights=parameter['weights'],
                                 p=parameter['p'],
                                 n_jobs=-1)
    model.fit(train_x, train_y)

    valid_prediction = model.predict_proba(valid_x)[:, 1]
    auc = metrics.roc_auc_score(valid_y, valid_prediction)
    return {'loss':-auc, 'status':STATUS_OK, 'model':model}
    # valid_prediction = model.predict(valid_x)
    # acc_score = metrics.accuracy_score(valid_y, valid_prediction)
    # return {'loss':-acc_score, 'status':STATUS_OK, 'model':model}

def hyperopt_my_xgb_regression(parameter):
    print(f"Trying parameters: {parameter}")
    model = KNeighborsRegressor(n_neighbors=parameter['n_neighbors'],
                                weights=parameter['weights'],
                                p=parameter['p'],
                                n_jobs=-1)
    model.fit(train_x, train_y)

    valid_prediction = model.predict(valid_x)
    r2 = metrics.r2_score(valid_y, valid_prediction)
    return {'loss':-r2, 'status':STATUS_OK, 'model':model}

trials = Trials()
if args.task_type=='classification':
    best = fmin(hyperopt_my_xgb_classification, space, algo=tpe.suggest, trials=trials, max_evals=30)
else:
    best = fmin(hyperopt_my_xgb_regression, space, algo=tpe.suggest, trials=trials, max_evals=30)
print(best)

args.n_neighbors = best['n_neighbors'] + 1  # 因为 hp.choice 从 0 开始索引
args.weights = ['uniform', 'distance'][best['weights']]
args.p = [1, 2, 3, 4, 5][best['p']]


result_pd = pd.DataFrame()
if args.task_type=='classification':
    result_pd['index'] = ['roc_auc','roc_prc','accuracy']
    for i in range(5):
        print(f'------time: {i+1}------')
        model = KNeighborsClassifier(n_neighbors=args.n_neighbors,
                                     weights=args.weights,
                                     p=args.p,
                                     n_jobs=-1)
        model.fit(train_x, train_y)
        train_prediction, valid_prediction, test_prediction = [model.predict_proba(x)[:, 1] for x in (train_x, valid_x, test_x)]
        prediction = pd.DataFrame({'smiles':test_smiles, 'pred':test_prediction, 'label':test_y})
        os.makedirs(f'./KNN_prediction/{args.fp_name}/', exist_ok=True)
        prediction.to_csv(f'./KNN_prediction/{args.fp_name}/KNN_{args.fp_name}_{args.data_name}_{i+1}_test_prediction', index=False) 
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
        model = KNeighborsRegressor(n_neighbors=args.n_neighbors,
                                    weights=args.weights,
                                    p=args.p,
                                    n_jobs=-1)
        model.fit(train_x, train_y)
        train_prediction, valid_prediction, test_prediction = [model.predict(x) for x in (train_x, valid_x, test_x)]
        prediction = pd.DataFrame({'smiles':test_smiles, 'pred':test_prediction, 'label':test_y})
        os.makedirs(f'./KNN_prediction/{args.fp_name}/', exist_ok=True)
        prediction.to_csv(f'./KNN_prediction/{args.fp_name}/KNN_{args.fp_name}_{args.data_name}_{i+1}_test_prediction', index=False)
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
os.makedirs(f'./KNN_result/{args.fp_name}/', exist_ok=True)
result_pd.to_csv(f'./KNN_result/{args.fp_name}/KNN_{args.fp_name}_{args.data_name}_all_result.csv', index=False)