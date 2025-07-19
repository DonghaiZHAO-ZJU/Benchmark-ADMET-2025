#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
This example trains multiple TabPFN models, which is computationally intensive.
"""
import os
import time
import argparse

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
    AutoTabPFNRegressor,
)

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

# Binary
# clf = AutoTabPFNClassifier(max_time=60 * 3)
# clf.fit(train_x, train_y)
# prediction_probabilities = clf.predict_proba(test_x)
# predictions = np.argmax(prediction_probabilities, axis=-1)

# print("ROC AUC:", roc_auc_score(test_y, prediction_probabilities[:, 1]))
# print("Accuracy", accuracy_score(test_y, predictions))

# Regression
reg = AutoTabPFNRegressor(max_time=60 * 3)
reg.fit(train_x, train_y)
predictions = reg.predict(test_x)
print("Mean Squared Error (MSE):", mean_squared_error(test_y, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(test_y, predictions))
print("R-squared (R^2):", r2_score(test_y, predictions))
