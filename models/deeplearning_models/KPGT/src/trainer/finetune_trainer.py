import os
import torch
import numpy as np
import pandas as pd
class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator0, evaluator1, evaluator2, result_tracker, summary_writer, device, model_name, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator0 = evaluator0
        self.evaluator1 = evaluator1
        self.evaluator2 = evaluator2
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
            
    def _forward_epoch(self, model, batched_data):
        (smiles, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        predictions = model.forward_tune(g, ecfp, md)
        return smiles, predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            _, predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)


    def fit(self, model, train_loader, val_loader, test_loader, loader=None, model_path=None):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_val_result1,best_test_result1,best_train_result1 = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_val_result2,best_test_result2,best_train_result2 = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0
        metric_list = []
        for epoch in range(1, self.args.n_epochs+1):
            print(f'{epoch}/{self.args.n_epochs+1}', flush=True)
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:
                val_result0, val_result1, val_result2 = self.eval(model, val_loader)
                test_result0, test_result1, test_result2 = self.eval(model, test_loader)
                train_result0, train_result1, train_result2 = self.eval(model, train_loader)
                if loader:
                    # fps_list = []
                    smiless = []
                    for batch_idx, batched_data in enumerate(loader):
                        (smiles_list, g, ecfp, md, labels) = batched_data
                        smiless.extend(smiles_list)
                        ecfp = ecfp.to(self.device)
                        md = md.to(self.device)
                        g = g.to(self.device)
                    #     fps = model.generate_fps(g, ecfp, md)
                    #     fps_list.extend(fps.detach().cpu().numpy().tolist())
                    # df = pd.DataFrame(fps_list) 
                    # df.insert(0, 'smiles', smiless)
                    # os.makedirs(f"../features/{self.args.dataset}/", exist_ok=True)
                    # df.to_csv(f"../features/{self.args.dataset}/{self.args.dataset}_KPGT_Embedding_{epoch}.csv", index=False)

                print(f'train {self.args.metric[0]}: {train_result0}, valid {self.args.metric[0]}: {val_result0}, test {self.args.metric[0]}: {test_result0}', flush=True)
                # if self.result_tracker.update(np.mean(best_val_result1), np.mean(val_result1)): # 针对不平衡的分类数据集
                if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result0)):
                    best_val_result, best_test_result, best_train_result = val_result0, test_result0, train_result0
                    best_val_result1, best_test_result1, best_train_result1 = val_result1, test_result1, train_result1
                    best_val_result2, best_test_result2, best_train_result2 = val_result2, test_result2, train_result2
                    best_epoch = epoch
                    if model_path is not None:
                        torch.save({'model_state_dict': model.state_dict()}, model_path)
                print("Training Results:")
                metrics={'epoch':epoch}
                for i, value in enumerate([train_result0, train_result1, train_result2]):
                    print(f"{self.args.metric[i]}: {value:.4f}")
                    metrics[f'train_{self.args.metric[i]}'] = value
                print("Validation Results:")
                for i, value in enumerate([val_result0, val_result1, val_result2]):
                    print(f"{self.args.metric[i]}: {value:.4f}")
                    metrics[f'valid_{self.args.metric[i]}'] = value
                print("Test Results:")
                for i, value in enumerate([test_result0, test_result1, test_result2]):
                    print(f"{self.args.metric[i]}: {value:.4f}")
                    metrics[f'test_{self.args.metric[i]}'] = value
                metric_list.append(metrics)
                # metric_df = pd.DataFrame(metric_list)
                # metric_df.to_csv(f"../features/{self.args.dataset}/{self.args.dataset}_metric.csv", index=False)
                if epoch - best_epoch >= 20:
                    print('break!')
                    break
        return best_train_result, best_val_result, best_test_result, best_train_result1, best_val_result1, best_test_result1, best_train_result2, best_val_result2, best_test_result2
    def eval(self, model, dataloader, out_path=None):
        model.eval()
        smiles_all = []
        predictions_all = []
        labels_all = []
        
        for batched_data in dataloader:
            smiles, predictions, labels = self._forward_epoch(model, batched_data)
            smiles_all.extend(smiles)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        result0 = self.evaluator0.eval(torch.cat(labels_all), torch.cat(predictions_all))
        result1 = self.evaluator1.eval(torch.cat(labels_all), torch.cat(predictions_all))
        result2 = self.evaluator2.eval(torch.cat(labels_all), torch.cat(predictions_all))
        if out_path is not None:
            if (self.label_mean is not None) and (self.label_std is not None):
                predictions_all = torch.cat(predictions_all).flatten()
                predictions_all = predictions_all*self.label_std.detach().cpu()+self.label_mean.detach().cpu()
                result_pd = pd.DataFrame({'smiles': smiles_all, 'pred':predictions_all.numpy(), 'label':torch.cat(labels_all).flatten().numpy()})
            else:
                if self.args.dataset_type == 'regression':
                    result_pd = pd.DataFrame({'smiles': smiles_all, 'pred':torch.cat(predictions_all).flatten().numpy(), 'label':torch.cat(labels_all).flatten().numpy()})
                else:
                    predictions_sigmoid = torch.sigmoid(torch.cat(predictions_all)).flatten().numpy()
                    result_pd = pd.DataFrame({'smiles': smiles_all, 'pred':predictions_sigmoid, 'label':torch.cat(labels_all).flatten().numpy()})
            result_pd.to_csv(out_path, index=False)
        return result0, result1, result2

    