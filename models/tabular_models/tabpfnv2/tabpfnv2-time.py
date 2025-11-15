import os
import time
import argparse

import pandas as pd
import numpy as np

from tabpfn import TabPFNClassifier, TabPFNRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

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
space = {'n_estimators':hp.choice('n_estimators', list(range(4,5))),
        #  'softmax_temperature': hp.choice('softmax_temperature', [i/10 for i in range(1,11)]),
        #  'average_before_softmax': hp.choice('average_before_softmax', [True, False]),
         }

process_data = pd.read_csv(f'./data/process_normalization_data/{args.task_name}.csv')
group_data = pd.read_csv(f'./data/data_group/{args.data_name}.csv')
merged_data = pd.merge(process_data, group_data, on=['smiles', args.task_name], how='inner')
columns_order = ['smiles', args.task_name, 'group'] + [col for col in merged_data.columns if col not in ['smiles', args.task_name, 'group']]
data = merged_data[columns_order]
features = data.columns.difference(['smiles', args.task_name, 'group'])


train_set = data[data['group']=='training']
valid_set = data[data['group']=='valid']
test_set = data[data['group']=='test']
print(len(train_set),len(valid_set),len(test_set))
train_x, train_y = train_set[features], train_set[args.task_name]
valid_x, valid_y = valid_set[features], valid_set[args.task_name]
test_x, test_y = test_set[features], test_set[args.task_name]
test_smiles = test_set['smiles']

def hyperopt_my_TabPFN_classification(parameter):
    model = TabPFNClassifier(n_estimators=parameter['n_estimators'], 
                            #  softmax_temperature=parameter['softmax_temperature'],
                            #  average_before_softmax=parameter['average_before_softmax'],
                             random_state=2044, n_jobs=-1)
    model.fit(train_x, train_y)

    valid_prediction = model.predict_proba(valid_x)[:, 1]
    auc = metrics.roc_auc_score(valid_y, valid_prediction)
    # auc = pr_auc_score(valid_y, valid_prediction)
    return {'loss':-auc, 'status':STATUS_OK, 'model':model}
    # valid_prediction = model.predict(valid_x)
    # acc_score = metrics.accuracy_score(valid_y, valid_prediction)
    # return {'loss':-acc_score, 'status':STATUS_OK, 'model':model}

def hyperopt_my_TabPFN_regression(parameter):
    model = TabPFNRegressor(n_estimators=parameter['n_estimators'], 
                            # softmax_temperature=parameter['softmax_temperature'],
                            # average_before_softmax=parameter['average_before_softmax'],
                            random_state=2044, n_jobs=-1)
    model.fit(train_x, train_y)

    valid_prediction = model.predict(valid_x)
    r2 = metrics.r2_score(valid_y, valid_prediction)
    return {'loss':-r2, 'status':STATUS_OK, 'model':model}

trials = Trials()
if args.task_type=='classification':
    best = fmin(hyperopt_my_TabPFN_classification, space, algo=tpe.suggest, trials=trials, max_evals=1)
else:
    best = fmin(hyperopt_my_TabPFN_regression, space, algo=tpe.suggest, trials=trials, max_evals=1)
print(best)

args.n_estimators = list(range(4,5))[best['n_estimators']]
# args.softmax_temperature = [i/10 for i in range(1,11)][best['softmax_temperature']]
# args.average_before_softmax = [True, False][best['average_before_softmax']]

result_pd = pd.DataFrame()

n_runs = 5  # 跑 5 次
training_times = []
testing_times = []
total_times = []


if args.task_type=='classification':
    result_pd['index'] = ['roc_auc','roc_prc','accuracy']
    for i in range(5):
        print(f'------time: {i+1}------')

        # 记录训练开始时间
        train_start = time.time()

        model = TabPFNClassifier(n_estimators=args.n_estimators,
                                #  softmax_temperature=args.softmax_temperature,
                                #  average_before_softmax=args.average_before_softmax,
                                 random_state=2024 + i * 10, n_jobs=-1)
        model.fit(train_x, train_y)

        # 记录训练结束时间
        train_end = time.time()
        training_time = train_end - train_start
        print(f"Training time: {training_time:.2f} seconds")

        # 记录测试开始时间
        test_start = time.time()

        test_prediction = model.predict_proba(test_x)[:, 1]

        # 记录测试结束时间
        test_end = time.time()
        testing_time = test_end - test_start
        print(f"Testing time: {testing_time:.2f} seconds")
        
        total_time = training_time + testing_time
        print(f"Total training and testing time: {total_time:.2f} seconds")

        # 保存当前运行的耗时
        training_times.append(training_time)
        testing_times.append(testing_time)

        # train_prediction, valid_prediction, test_prediction = [model.predict_proba(x)[:, 1] for x in (train_x, valid_x, test_x)]
        # prediction = pd.DataFrame({'smiles':test_smiles, 'pred':test_prediction, 'label':test_y})
        # os.makedirs(f'./TabPFN2_prediction_no_tiaocan_rdkit/{args.fp_name}/', exist_ok=True)
        # prediction.to_csv(f'./TabPFN2_prediction_no_tiaocan_rdkit/{args.fp_name}/{args.data_name}_{i+1}.csv', index=False) 
        # roc_auc_score = metrics.roc_auc_score(train_y, train_prediction), metrics.roc_auc_score(valid_y, valid_prediction), metrics.roc_auc_score(test_y, test_prediction)
        # roc_prc_score = pr_auc_score(train_y, train_prediction), pr_auc_score(valid_y, valid_prediction), pr_auc_score(test_y, test_prediction)    
        # acc_score = metrics.accuracy_score(train_y, list(map(round, train_prediction))), metrics.accuracy_score(valid_y, list(map(round, valid_prediction))), metrics.accuracy_score(test_y, list(map(round, test_prediction)))
        # result_pd['train_' + str(i+1)] = [roc_auc_score[0], roc_prc_score[0], acc_score[0]]
        # result_pd['val_' + str(i+1)] = [roc_auc_score[1], roc_prc_score[1], acc_score[1]]
        # result_pd['test_' + str(i+1)] = [roc_auc_score[2], roc_prc_score[2], acc_score[2]]
        # print(result_pd)                  
else:
    result_pd['index'] = ['r2','rmse','mae']
    for i in range(5):
        print(f'------time: {i+1}------')
        model = TabPFNRegressor(n_estimators=args.n_estimators,
                                # softmax_temperature=args.softmax_temperature,
                                # average_before_softmax=args.average_before_softmax,
                                random_state=2024 + i * 10, n_jobs=-1)
        model.fit(train_x, train_y)
        train_prediction, valid_prediction, test_prediction = [model.predict(x) for x in (train_x, valid_x, test_x)]
        prediction = pd.DataFrame({'smiles':test_smiles, 'pred':test_prediction, 'label':test_y})
        os.makedirs(f'./TabPFN2_prediction_no_tiaocan1_rdkit/{args.fp_name}/', exist_ok=True)
        prediction.to_csv(f'./TabPFN2_prediction_no_tiaocan1_rdkit/{args.fp_name}/{args.data_name}_{i+1}.csv', index=False)
        r2_score = metrics.r2_score(train_y, train_prediction), metrics.r2_score(valid_y, valid_prediction), metrics.r2_score(test_y, test_prediction)
        rmse_score = np.sqrt(metrics.mean_squared_error(train_y, train_prediction)), np.sqrt(metrics.mean_squared_error(valid_y, valid_prediction)), np.sqrt(metrics.mean_squared_error(test_y, test_prediction))
        mae_score = metrics.mean_absolute_error(train_y, train_prediction), metrics.mean_absolute_error(valid_y, valid_prediction), metrics.mean_absolute_error(test_y, test_prediction)
        result_pd['train_' + str(i+1)] = [r2_score[0], rmse_score[0], mae_score[0]]
        result_pd['val_' + str(i+1)] = [r2_score[1], rmse_score[1], mae_score[1]]
        result_pd['test_' + str(i+1)] = [r2_score[2], rmse_score[2], mae_score[2]]
        print(result_pd)

# 创建 DataFrame
data = {
    "Model": ['TabPFN'],
    **{f"train_{i+1}": [training_times[i]] for i in range(5)},
    **{f"test_{i+1}": [testing_times[i]] for i in range(5)}
}

df = pd.DataFrame(data)

# 显示 DataFrame
print(df)

# 保存为 CSV
df.to_csv("time/TabPFN.csv")

elapsed = (time.time() - start)
m, s = divmod(elapsed, 60)
h, m = divmod(m, 60)
print("{} {} time used:, {:d}:{:d}:{:d}".format(args.data_name, args.fp_name, int(h), int(m), int(s)), flush=True)

# result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
# result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
# result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
# result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
# result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
# result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
# os.makedirs(f'./TabPFN2_result_no_tiaocan_rdkit/{args.fp_name}/', exist_ok=True)
# result_pd.to_csv(f'./TabPFN2_result_no_tiaocan_rdkit/{args.fp_name}/TabPFN_{args.fp_name}_{args.data_name}_all_result.csv', index=False)
