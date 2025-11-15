import os
import json
import random
import platform
import pathlib
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from utils.features import gen_features
from utils.configurator import Configurator
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer

import torch as th
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, auc

def set_random_seed(seed=10):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)

set_random_seed(0)

def calculate_prauc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return pr_auc

prauc_scorer = make_scorer(
    name='pr_auc',
    score_func=calculate_prauc,
    optimum=1,
    greater_is_better=True,
    needs_threshold=True
)

def build_df(train_data_path: str):
    _, sfx = os.path.splitext(train_data_path)
    if sfx == '.csv':
        df_train = pd.read_csv(train_data_path)
    else:
        df_train = pd.read_excel(train_data_path)
    return df_train

def smis2fps(smis: list, features='rdkit_2d_norm'):
    fps = []
    # 使用 tqdm 包装循环以显示进度条
    for smi in tqdm(smis, desc="Processing SMILES", unit="molecule"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:  # 检查分子是否有效
            fp = gen_features(mol, features=features)
            fps.append(fp)
        else:
            fps.append(np.nan)  # 如果 SMILES 无效，添加 NaN
    return np.array(fps)

def gen_fps_df(data: pd.DataFrame, data_with_descriptors=pd.DataFrame):
    # smiless = list(data['smiles'])
    # fps_np = smis2fps(smiless, features=feature)
    # df_fps_np = pd.DataFrame(fps_np)
    # df_fps_np_label = pd.concat([df_fps_np, data[[i for i in data.columns if i not in ['smiles', 'group']]]], axis=1)
    # df_fps_np_label.columns = list(df_fps_np_label.columns[:-1]) + ['label']
    merged_df = pd.merge(data, data_with_descriptors, on='smiles', how='inner')
    useful_columns = [i for i in merged_df.columns if i not in ['smiles']]
    return merged_df[useful_columns]


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model using AutoGluon.")
    
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--split_method", type=str, required=True)
    parser.add_argument("--split_seed", type=str, required=True)
    parser.add_argument("--eval_metric", type=str)
    parser.add_argument("--time_limit", type=int, default=3600)

    args = parser.parse_args()
    print(args)

    task, split_method, split_seed = args.task, args.split_method, args.split_seed
    if os.path.exists(f"datasets/processed_data/{task}.csv"):
        print(f'descriptors has existed!')
        data_with_descriptors = pd.read_csv(f"datasets/processed_data/{task}.csv")
    else:
        print(f'descriptors has not existed!')
        data = pd.read_csv(f"./datasets/origin_data/{task}.csv")
        smiless = list(data['smiles'])
        descriptors = smis2fps(smiless)
        data_with_descriptors = pd.DataFrame(descriptors)
        data_with_descriptors.insert(0, 'smiles', smiless)
        data_with_descriptors.to_csv(f"./datasets/processed_data/{task}.csv", index=False)
    train_data_path = f"./datasets/data_with_group/{task}_{split_method}_{split_seed}/{task}_{split_method}_{split_seed}_training.csv"
    valid_data_path = f"./datasets/data_with_group/{task}_{split_method}_{split_seed}/{task}_{split_method}_{split_seed}_valid.csv"
    test_data_path = f"./datasets/data_with_group/{task}_{split_method}_{split_seed}/{task}_{split_method}_{split_seed}_test.csv"
    model_dir = f"./model/{task}_{split_method}_{split_seed}/"
    train_data, valid_data, test_data = build_df(train_data_path), build_df(valid_data_path), build_df(test_data_path)
    df_train_fps_label, df_valid_fps_label, df_test_fps_label = gen_fps_df(train_data, data_with_descriptors), gen_fps_df(valid_data, data_with_descriptors), gen_fps_df(test_data, data_with_descriptors)
    predictor = TabularPredictor(
        label=task, 
        path=model_dir, 
        eval_metric= prauc_scorer if args.eval_metric=="pr_auc" else args.eval_metric,
        ).fit(
        train_data=df_train_fps_label,
        tuning_data=df_valid_fps_label,
        time_limit=args.time_limit,
        # presets=["optimize_for_deployment"],
    )
    # predictor = TabularPredictor.load('/root/data1/admet_models_validation/automl/model/HalfLife_random_2024')
    if predictor.can_predict_proba:
        print('Your task is classification.')
        perf = predictor.evaluate(df_test_fps_label)
        perf_df = pd.DataFrame.from_dict(perf, orient='index').reset_index()
        perf_df.columns = ['Metric', 'Value']
        df_test_fps = df_test_fps_label.drop(columns=[task])
        
        preds_df = predictor.predict_proba(df_test_fps)
        df_pred_smis = pd.concat([test_data['smiles'], preds_df.iloc[:,1].rename('pred'), df_test_fps_label[task]], axis=1)
        df_pred_smis.to_csv(f"./prediction/{task}_{split_method}_{split_seed}.csv", index=False)
        pr_auc = calculate_prauc(df_test_fps_label[task], preds_df.iloc[:,1].rename('pred'))
        new_row = pd.DataFrame({'Metric': ['pr_auc'], 'Value': [pr_auc]})
        perf_df = pd.concat([perf_df.iloc[:1], new_row, perf_df.iloc[1:]]).reset_index(drop=True)
    else:
        print('Your task is regression.')
        perf = predictor.evaluate(df_test_fps_label)
        perf['mean_absolute_error'] = -perf['mean_absolute_error']
        perf['root_mean_squared_error'] = -perf['root_mean_squared_error']
        perf['mean_squared_error'] = -perf['mean_squared_error']
        perf['median_absolute_error'] = -perf['median_absolute_error']
        perf_df = pd.DataFrame.from_dict(perf, orient='index').reset_index()
        perf_df.columns = ['Metric', 'Value']
        df_test_fps = df_test_fps_label.drop(columns=[task])

        preds_df = predictor.predict(df_test_fps)
        print(type(preds_df))
        df_pred_smis = pd.concat([test_data['smiles'], preds_df.rename('pred'), df_test_fps_label[task]], axis=1)
        df_pred_smis.to_csv(f"./prediction/{task}_{split_method}_{split_seed}.csv", index=False)
    perf_df.to_csv(f'./results/{task}_{split_method}_{split_seed}.csv', index=False)
    print(perf_df)

    
if __name__ == '__main__':
    main()