import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import  roc_auc_score,accuracy_score, precision_recall_curve, auc, \
                             precision_score, recall_score, matthews_corrcoef, r2_score, mean_absolute_error

def metric(label,output,task_type):
    if task_type=='classification':
        zs = torch.sigmoid(output).to('cpu').data.numpy()
        # print(zs.shape)
        ts = label.to('cpu').data.numpy()
        # print(ts.shape)
        preds = list(map(lambda x: (x >= 0.5).astype(int), zs))

        rocauc = roc_auc_score(ts, zs)
        precision, recall, _thresholds = precision_recall_curve(ts, zs)
        prauc = auc(recall, precision)
        preds_list, t_list = [], []
        preds_list = np.append(preds_list, preds)
        t_list = np.append(t_list, ts)
        acc = accuracy_score(t_list, preds_list)
        precision = precision_score(t_list, preds_list)

        recall = recall_score(t_list, preds_list)
        mcc = matthews_corrcoef(t_list, preds_list)

        f1_score = (2 * precision * recall) / (recall + precision)
        return rocauc,prauc,acc,precision,recall,f1_score,mcc
    else:
        zs = output.to('cpu').data.numpy()
        ts = label.to('cpu').data.numpy()
        r2 = r2_score(ts, zs)
        rmse = np.sqrt(F.mse_loss(torch.tensor(zs), torch.tensor(ts)).cpu().item())
        mae = mean_absolute_error(ts, zs)
        return r2,rmse,mae



