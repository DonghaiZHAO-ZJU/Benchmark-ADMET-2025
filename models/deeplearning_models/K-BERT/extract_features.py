import tqdm
import pandas as pd
import numpy as np
from experiment.build_data import construct_input_from_smiles
from experiment.atom_embedding_generator import bert_atom_embedding
task_list = ['CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50', 
            'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 
            'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki', 
            'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 
            'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 
            'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 
            'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 
            'CHEMBL4616_EC50', 'CHEMBL4792_Ki']
for task_name in task_list:
    print(task_name)
    dataset = pd.read_csv('./origin_data/'+task_name+'.csv')
    smiles_list = dataset['smiles'].tolist()
    pretrain_features_list = []
    for i, smiles in tqdm.tqdm(enumerate(smiles_list), total=len(smiles_list)):
        try:
            h_global, g_atom = bert_atom_embedding(smiles, pretrain_model='pretrain_k_bert_wcl_epoch_7.pth')
            pretrain_features_list.append(h_global)
        except:
            pretrain_features_list.append(['NaN' for x in range(768)])
    np.savez_compressed(f"features/k_bert_{task_name}.npz", fps=np.array(pretrain_features_list))
    df = pd.DataFrame(pretrain_features_list)
    df.insert(0, 'smiles', smiles_list)
    df.to_csv(f"features/{task_name}_K_BERT_Embedding.csv", index=False)