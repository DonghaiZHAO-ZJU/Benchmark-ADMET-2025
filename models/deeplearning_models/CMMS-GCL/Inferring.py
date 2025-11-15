import sys
from model import *
from utils import *
from evalution import *
import torch.nn.functional as F
import pandas as pd
import time


def predicting(data_name, task_type, run, model, device, loader, mean, std):
    data_origin = pd.read_csv(f'./data/raw/{data_name}/{data_name}_test.csv')
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
        model_file_name = f'./model/{args.data_name}_{run+1}.pt'
        model.load_state_dict(torch.load(model_file_name))
        test_results = predicting(args.data_name, args.task_type, run, model, device, test_loader, mean, std)
        print(test_results)
