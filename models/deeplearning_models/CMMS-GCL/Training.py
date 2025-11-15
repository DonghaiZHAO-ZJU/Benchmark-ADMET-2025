import sys
from model import *
from utils import *
from evalution import *
import torch.nn.functional as F
import pandas as pd
import time


def ViewContrastiveLoss(view_i, view_j, batch,temperature):

    z_i = F.normalize(view_i, dim=1)
    z_j = F.normalize(view_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                            dim=2)
    similarity_matrix = similarity_matrix.to(device)
    sim_ij = torch.diag(similarity_matrix, batch)
    sim_ji = torch.diag(similarity_matrix, -batch)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temperature)
    negatives_mask = torch.ones(2 * batch, 2 * batch) - torch.eye(2 * batch, 2 * batch)
    negatives_mask = negatives_mask.to(device)
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss = torch.sum(loss_partial) / (2 * batch)

    return loss

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, mean, std):
    print('Training on {} samples...'.format(len(train_loader.dataset)), flush=True)
    model.train()
    total_loss = 0
    N = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        n = data.y.shape[0]  # batch
        optimizer.zero_grad()
        output,x_g,y_g= model(data,data.x,data.edge_index,data.batch,data.smi_em)
        if (mean is not None) and (std is not None):
            labels = standardization_np(data.y, mean, std)
        else:
            labels = data.y
        loss_1 = criterion(output, labels)
        T = 0.2
        loss_2 = ViewContrastiveLoss (x_g,y_g,n,T)
        loss = loss_1 + 0.1*loss_2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*n
        N += n

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()), flush=True)
    return total_loss / N    

def predicting(model, device, loader, mean, std, data_name=None, task_type=None, run=None):
    
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)), flush=True)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, x_g, y_g = model(data, data.x, data.edge_index, data.batch, data.smi_em)
            if (mean is not None) and (std is not None):
                output = re_standar_np(output, mean, std)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)
    if data_name is not None and task_type is not None and run is not None:
        data_origin = pd.read_csv(f'data/raw/{data_name}/{data_name}_test.csv')
        if task_type=="classification":
            total_preds1 = torch.sigmoid(total_preds)
            prediction = pd.DataFrame({'smiles': data_origin['smiles'].values, 'pred':total_preds1.flatten(), 'label':total_labels.flatten()})
        else:
            prediction = pd.DataFrame({'smiles': data_origin['smiles'].values, 'pred':total_preds.flatten(), 'label':total_labels.flatten()})
        os.makedirs('./prediction/', exist_ok=True)
        prediction.to_csv(f'./prediction/CMMSGCL_{data_name}_{run+1}_test_prediction.csv', index=False)
    return metric(total_labels,total_preds,args.task_type)

import random
def set_random_seed(seed=0, cuda_deterministic=True):
    """
    This function is only used for reproducbility, 
    DDP model doesn't need to use same seed for model initialization, 
    as it will automatically send the initialized model from master node to other nodes. 
    Notice this requires no change of model after call DDP(model)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, help="Name of the data")
parser.add_argument("--task_type", type=str, choices=['classification', 'regression'], help="type of the data")
args = parser.parse_args()

cuda_name = "cuda:0"
# if len(sys.argv) > 3:
#     cuda_name = "cuda:" + str(int(sys.argv[4]))
# print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 500
PATIENCE = 30

print('Learning rate: ', LR, flush=True)
print('Epochs: ', NUM_EPOCHS, flush=True)

processed_train = os.path.join('data/processed/', args.data_name, f'{args.data_name}_train.pt')
processed_valid = os.path.join('data/processed/', args.data_name, f'{args.data_name}_valid.pt')
processed_test = os.path.join('data/processed/', args.data_name, f'{args.data_name}_test.pt')

if ((not os.path.isfile(processed_train)) or (not os.path.isfile(processed_valid)) or (not os.path.isfile(processed_test))):
        print('please run create_data.py to prepare data in pytorch format!', flush=True)
else:
    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []
    result_pd = pd.DataFrame()
    if args.task_type=='classification':
        metric_name = 'roc_auc'
        result_pd['index'] = ['roc_auc', 'roc_prc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc']
    else:
        metric_name = 'r2'
        result_pd['index'] = ['r2', 'rmse', 'mae']
    start = time.time()
    for run in range(5):
        seed = 2024+run*10
        print(f'Current seed: {seed}')
        set_random_seed(seed)
        train_data = TestbedDataset(root='data', dataset='train', data_name=args.data_name)
        valid_data = TestbedDataset(root='data', dataset='valid', data_name=args.data_name)
        test_data = TestbedDataset(root='data', dataset='test', data_name=args.data_name)
        if args.task_type=='regression':
            mean, std = train_data.calc_scaler()
            print(f'trainset mean:{mean} std:{std}')
        else:
            mean, std = None, None

        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = CMMS_GCL().to(device)
        if args.task_type=='classification':
            criterion = nn.BCEWithLogitsLoss(weight=train_data.label_weight().to(device))
        else:
            criterion = nn.MSELoss()
        contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_valid_score = float('-inf')

        model_file_name = f'{args.data_name}_{run+1}.pt'
        result_file_name = 'result' + '.csv'
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, device, train_loader, optimizer, epoch + 1, mean, std)
            train_results = predicting(model, device, train_loader, mean, std)
            valid_results = predicting(model, device, valid_loader, mean, std)
            test_results = predicting(model, device, test_loader, mean, std)
            if valid_results[0] > best_valid_score:
                best_train_score = train_results[0]
                best_valid_score = valid_results[0]
                best_test_score = test_results[0]
                best_train_results = train_results
                best_valid_results = valid_results
                best_test_results = test_results
                best_epoch = epoch
                test_results = predicting(model, device, test_loader, mean, std, data_name=args.data_name, task_type=args.task_type, run=run)
                torch.save(model.state_dict(), os.path.join('./model', model_file_name))
                count=0
            else:
                count+=1
                print(f'EarlyStopping counter: {count} out of {PATIENCE}', flush=True)

            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Metric: {metric_name}, Train: {train_results[0]:.4f}, Val: {valid_results[0]:.4f}, best Val: {best_valid_score:.4f}, Test: {test_results[0]:.4f}', flush=True)
            
            if (epoch-best_epoch) >= PATIENCE:
                print('out of patience!', flush=True)
                break
        
        result_pd['train_'+str(run+1)] = best_train_results
        result_pd['val_'+str(run+1)] = best_valid_results
        result_pd['test_'+str(run+1)] = best_test_results
        print('******{}, {}th_time_result******'.format(args.data_name, run + 1), flush=True)
        print("train_result:", round(best_train_score, 4), flush=True)
        print("val_result:", round(best_valid_score, 4), flush=True)
        print("test_result:", round(best_test_score, 4), flush=True)
        all_times_train_result.append(best_train_score)
        all_times_val_result.append(best_valid_score)
        all_times_test_result.append(best_test_score)

    print("************************************{}_times_result************************************".format(run + 1), flush=True)
    print('the train result of all tasks ({}): '.format(metric_name), np.array(all_times_train_result), flush=True)
    print('the average train result of all tasks ({}): {:.4f}'.format(metric_name, np.array(all_times_train_result).mean()), flush=True)
    print('the train result of all tasks (std): {:.4f}'.format(np.array(all_times_train_result).std()), flush=True)
    print('the train result of all tasks (var): {:.4f}'.format(np.array(all_times_train_result).var()), flush=True)

    print('the val result of all tasks ({}): '.format(metric_name), np.array(all_times_val_result), flush=True)
    print('the average val result of all tasks ({}): {:.4f}'.format(metric_name, np.array(all_times_val_result).mean()), flush=True)
    print('the val result of all tasks (std): {:.4f}'.format(np.array(all_times_val_result).std()), flush=True)
    print('the val result of all tasks (var): {:.4f}'.format(np.array(all_times_val_result).var()), flush=True)

    print('the test result of all tasks ({}):'.format(metric_name), np.array(all_times_test_result), flush=True)
    print('the average test result of all tasks ({}): {:.4f}'.format(metric_name, np.array(all_times_test_result).mean()), flush=True)
    print('the test result of all tasks (std): {:.4f}'.format(np.array(all_times_test_result).std()), flush=True)
    print('the test result of all tasks (var): {:.4f}'.format(np.array(all_times_test_result).var()), flush=True)

    elapsed = (time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("{} time used:, {:d}:{:d}:{:d}".format(args.data_name, int(h), int(m), int(s)), flush=True)
    
    result_pd['train_mean'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].mean(axis=1).round(4)
    result_pd['train_std'] = result_pd[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].std(axis=1).round(4)
    result_pd['val_mean'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].mean(axis=1).round(4)
    result_pd['val_std'] = result_pd[['val_1', 'val_2', 'val_3', 'val_4', 'val_5']].std(axis=1).round(4)
    result_pd['test_mean'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].mean(axis=1).round(4)
    result_pd['test_std'] = result_pd[['test_1', 'test_2', 'test_3', 'test_4', 'test_5']].std(axis=1).round(4)
    result_pd.to_csv('./result/CMMSGCL_' + args.data_name + '_all_result.csv', index=False)
