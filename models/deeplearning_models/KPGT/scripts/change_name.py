import pandas as pd
import numpy as np

# single smiles unimol representation
# clf = UniMolRepr(data_type='molecule', remove_hs=True)
for task in ['CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50', 
            'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 
            'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki', 
            'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50', 
            'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki', 
            'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 
            'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 
            'CHEMBL4616_EC50', 'CHEMBL4792_Ki']:
    
    origin_data = pd.read_csv(f'../datasets/{task}/{task}.csv')
    smiles = origin_data['smiles']  # 获取 smiles 列

    # 这里假设你已经获得了 unimol_repr
    # np.savez_compressed(f"features/unimol_{task}.npz", fps=np.array(unimol_repr['cls_repr']))

    data = np.load(f"../datasets/{task}/kpgt_base_{task}.npz")
    # 检查是否有 embedding 内所有元素都相同
    fps = data['fps']
    
    # 将 fps 转换为 DataFrame
    df = pd.DataFrame(fps)

    # 插入 smiles 列到第一列
    df.insert(0, 'smiles', smiles)  # loc=0 表示插入到第一列

    # 保存 DataFrame 到 CSV 文件
    df.to_csv(f"../features/{task}_KPGT_Embedding.csv", index=False)