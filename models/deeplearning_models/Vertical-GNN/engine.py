import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd

def standardization_np(data, mean, std):
    return (data - mean) / (std + 1e-10)
def re_standar_np(data, mean, std):
    return data * (std + 1e-10) + mean

class Engine_regression:
    def __init__(self, model, optimizer, mean, std, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.mean = mean
        self.std = std

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.MSELoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
            labels = standardization_np(data.y.unsqueeze(1), self.mean, self.std)            
            loss = self.loss_fn(labels, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()

        return final_loss / len(data_loader)

    def validate(self, data_loader, out_path=None):
        self.model.eval()
        smiles_list = []
        preds = []
        true_labels = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(
                    data.x, data.edge_attr, data.edge_index, data.batch
                )
                outputs = re_standar_np(outputs, self.mean, self.std)
                smiles_list.append(data.smiles)
                preds.append(outputs.to("cpu").detach().numpy())
                true_labels.append(data.y.unsqueeze(1).to("cpu").detach().numpy())
            smiles_list = np.concatenate(smiles_list, axis=0)
            preds = np.concatenate(preds, axis=0)
            true_labels = np.concatenate(true_labels, axis=0)
            r2 = r2_score(true_labels, preds)
            rmse = mean_squared_error(true_labels, preds) ** 0.5
            mae = mean_absolute_error(true_labels, preds)
        if out_path:
            output = pd.DataFrame()
            output['smiles'] = smiles_list
            output['pred'] = preds
            output['label'] = true_labels
            output.to_csv(out_path, index=False)

        return r2, rmse, mae


class Engine_class:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
            loss = self.loss_fn(data.y.unsqueeze(1), outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    def validate(self, data_loader, out_path=None):
        self.model.eval()
        smiles_list = []
        preds = []
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(
                    data.x, data.edge_attr, data.edge_index, data.batch
                )
                smiles_list.append(data.smiles)
                preds.append(torch.sigmoid(outputs).to("cpu").detach().numpy())
                pred_labels.append(torch.round(torch.sigmoid(outputs)).to("cpu").detach().numpy())
                true_labels.append(data.y.unsqueeze(1).to("cpu").detach().numpy())
            smiles_list = np.concatenate(smiles_list, axis=0)
            preds = np.concatenate(preds, axis=0)
            pred_labels = np.concatenate(pred_labels, axis=0)
            true_labels = np.concatenate(true_labels, axis=0)
            rocauc = roc_auc_score(true_labels, preds)
            precision, recall, _ = precision_recall_curve(true_labels, preds)
            prauc = auc(recall, precision)
            acc = accuracy_score(true_labels, pred_labels)
        if out_path:
            output = pd.DataFrame()
            output['smiles'] = smiles_list
            output['pred'] = preds
            output['label'] = true_labels
            output.to_csv(out_path, index=False)
         
        return rocauc, prauc, acc
    
    def get_embeds(self, data_loader):
        embeddings_list = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs, embeddings = self.model(
                    data.x, data.edge_attr, data.edge_index, data.batch
                        )
                embeddings_list.append(embeddings)
        return embeddings_list
                


class EngineHOB_no_edge:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data.x, data.edge_index, data.batch)
            loss = self.loss_fn(data.y.unsqueeze(1), outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1), outputs)
                final_loss += loss.item()
        return final_loss / len(data_loader)

    def test(self, data_loader):
        self.model.eval()
        final_loss = 0
        acc_total = 0
        f1_total = 0
        roc_auc_total = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1), outputs)
                final_loss += loss.item()

                acc = accuracy_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.round(torch.sigmoid(outputs)).to("cpu").detach().numpy(),
                )
                acc_total += acc

                f1 = f1_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.round(torch.sigmoid(outputs)).to("cpu").detach().numpy(),
                )
                f1_total += f1

                roc_auc = roc_auc_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.sigmoid(outputs).to("cpu").detach().numpy(),
                )
                roc_auc_total += roc_auc

        return (
            final_loss / len(data_loader),
            acc_total / len(data_loader),
            f1_total / len(data_loader),
            roc_auc_total / len(data_loader),
        )
