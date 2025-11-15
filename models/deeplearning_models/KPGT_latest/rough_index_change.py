import rogi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rogi import RoughnessIndex, SARI, MODI, RMODI

from sklearn.manifold import MDS
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.interpolate import griddata
from scipy.spatial.distance import cosine
import seaborn as sns
from matplotlib import cm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

dataset_names = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability","Caco2","HalfLife","VDss","PAMPA1"]
classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability"]
regression_tasks = ["Caco2","HalfLife","VDss","PAMPA1"]


def calculate_tanimoto(smiless, fp_type='morgan'):
    n = len(smiless)
    sim_matrix = np.zeros((n, n))
    if fp_type=='morgan':
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), nBits=2048, radius=2, useChirality=True) for smiles in smiless]
    if fp_type=='MACCSkeys':
        fps = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles)) for smiles in smiless]
    
    # compute similarity matrix
    sim_matrix = np.zeros(shape=(len(smiless), len(smiless)))
    for i in range(len(smiless)):
        # i+1 becauase we know the diagonal is zero
        sim_matrix[i, i + 1:] = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
        sim_matrix[i + 1:, i] = sim_matrix[i, i + 1:]
    return sim_matrix

def extract_upper_tri(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]

def calculate_cosine(embeddings):
    pairwise_dists = pdist(embeddings, 'cosine')
    return 1 - squareform(pairwise_dists)

def calculate_cosine_distance(embeddings):
    pairwise_dists = pdist(embeddings, 'cosine')
    return squareform(pairwise_dists)

def plot3d(Dx, Y, ax, prop_label="", rccounts=100):

    print("projecting on 2D plane...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    X_2d = mds.fit_transform(squareform(Dx))

    # get x,y,z for 3D plot
    x = X_2d[:, 0]
    y = X_2d[:, 1]
    z = Y

    # property max/min
    vmin = np.min(z)
    vmax = np.max(z)

    # get interpolation
    dx = 1000j
    grid_x, grid_y = np.mgrid[x.min():x.max():dx, y.min():y.max():dx]
    grid_z = griddata(X_2d, z, (grid_x, grid_y), method='linear', rescale=True)

    # plot in 3D
    print("plotting...")
    masked_grid_z = np.ma.masked_invalid(grid_z)

    cmap = cm.get_cmap("coolwarm")
    colors = (masked_grid_z - vmin) / (vmax-vmin)
    facecolors = cmap(colors)

    ax.plot_surface(grid_x, grid_y, masked_grid_z, rcount=rccounts, ccount=rccounts, 
                    facecolors=facecolors, alpha=0.75, linewidth=0, zorder=10,
                    vmin=vmin, vmax=vmax)


    ax.set_xlabel(r'$z_1$', labelpad=8)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel(r'$z_2$', labelpad=8)
    ax.set_ylim(y.min(), y.max())
    ax.set_zlabel(rf'${prop_label}$', labelpad=8)
    ax.set_zlim(z.min(), z.max())
    ax.set_box_aspect(aspect=None, zoom=0.8) # 用来解决z轴标签显示不全的问题

    ax.view_init(45, -60)

    ax.dist = 11

    return X_2d

# task = 'VDss'
# print(task)
# data = pd.read_csv(f'datasets/{task}/{task}.csv')
# metrics = pd.read_csv(f'features/{task}/{task}_metric.csv')
# roughness = []
# for i in range(41):
#     print(f'epoch: {i}')
#     if i==0:
#         embeddings = pd.read_csv(f'features/{task}_KPGT_Embedding.csv')
#     else:
#         embeddings = pd.read_csv(f'features/{task}/{task}_KPGT_Embedding_{i}.csv')
#     merged_data = pd.merge(embeddings, data, on=['smiles'], how='inner')
#     features = merged_data.columns.difference(['smiles', task, 'group'])
#     ri = RoughnessIndex(Y=merged_data[task], X=merged_data[features], metric='cosine')
#     rogi_score = ri.compute_index()
#     print(f"ROGI = {rogi_score}")
#     if i>0:
#         roughness.append(rogi_score)
#     fig = plt.figure(figsize=(7,5), dpi=600)
#     ax = fig.add_subplot(projection = '3d')
#     X_2d = plot3d(Dx=ri._Dx, Y=merged_data[task], ax=ax, prop_label=task, rccounts=1000)  # use rccounts=1000 for high res images
#     plt.tight_layout()
#     fig.savefig(f"graphs/3d-landscape-{task}-KPGT_Embedding_epoch_{i}.png", dpi=1200, bbox_inches='tight')
# metrics['ROGI'] = roughness
# metrics.to_csv(f'features/{task}/{task}_metric_with_ROGI.csv', index=False)

print("SARI改成动态阈值,RMODI调整")
print("SARI")
for task in ['Caco2', 'HalfLife', 'VDss']:
    print(task)
    data = pd.read_csv(f'datasets/{task}/{task}.csv')
    smiless = data['smiles'].tolist()
    MACCSkeys_similarity = extract_upper_tri(calculate_tanimoto(smiless, "MACCSkeys"))
    metrics = pd.read_csv(f'features/{task}/{task}_metric.csv')
    roughness = []
    if task == 'Caco2':
        epochs = 31
    if task == 'HalfLife':
        epochs = 32
    if task == 'VDss':
        epochs = 41
    for i in range(epochs):
        print(f'epoch: {i}')
        if i==0:
            embeddings = pd.read_csv(f'features/{task}_KPGT_Embedding.csv')
        else:
            embeddings = pd.read_csv(f'features/{task}/{task}_KPGT_Embedding_{i}.csv')
        merged_data = pd.merge(embeddings, data, on=['smiles'], how='inner')
        features = merged_data.columns.difference(['smiles', task, 'group'])
        smiless = merged_data['smiles'].values
        X=merged_data[features].values
        Embeddings_similarity_matrix = calculate_cosine(X)
        if i==0:
            Embeddings_similarity = extract_upper_tri(Embeddings_similarity_matrix)
            p = np.mean(MACCSkeys_similarity > 0.6)
            t_embeddings = np.percentile(Embeddings_similarity, 100 * (1 - p))    
        print(f"MACCS相似比例: {p:.2%}")
        print(f"Embeddings等效阈值: {t_embeddings:.4f}")            
        sari = SARI(pKi=merged_data[task], sim_matrix=Embeddings_similarity_matrix)
        sari_score, raw_cont, raw_disc = sari.compute_sari(similarity_threshold=t_embeddings)
        print(f"SARI = {sari_score}, raw_cont = {raw_cont}, raw_disc = {raw_disc}")
        if i>0:
            roughness.append(sari_score)
    metrics['SARI'] = roughness
    metrics.to_csv(f'features/{task}/{task}_metric_with_SARI(Dynamic Thresholds).csv', index=False)


print("MODI")
for task in ['Caco2', 'HalfLife', 'VDss']:
    print(task)
    data = pd.read_csv(f'datasets/{task}/{task}.csv')
    metrics = pd.read_csv(f'features/{task}/{task}_metric.csv')
    roughness = []
    if task == 'Caco2':
        epochs = 31
    if task == 'HalfLife':
        epochs = 32
    if task == 'VDss':
        epochs = 41
    for i in range(epochs):
        print(f'epoch: {i}')
        if i==0:
            embeddings = pd.read_csv(f'features/{task}_KPGT_Embedding.csv')
        else:
            embeddings = pd.read_csv(f'features/{task}/{task}_KPGT_Embedding_{i}.csv')
        merged_data = pd.merge(embeddings, data, on=['smiles'], how='inner')
        features = merged_data.columns.difference(['smiles', task, 'group'])
        X=merged_data[features].values
        # rogi_score = ri.compute_index()
        if task in classification_tasks:
            modi_score = MODI(Dx=calculate_cosine_distance(X), Y=merged_data[task].tolist())
            print(f"RMODI = {modi_score}")
        else:
            modi_score = RMODI(Dx=calculate_cosine_distance(X), Y=merged_data[task].tolist())
            print(f"RMODI = {modi_score}")
        if i>0:
            roughness.append(modi_score)
    metrics['MODI'] = roughness
    metrics.to_csv(f'features/{task}/{task}_metric_with_MODI.csv', index=False)

