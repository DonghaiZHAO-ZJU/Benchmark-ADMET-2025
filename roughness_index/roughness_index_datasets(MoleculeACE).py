import rogi
import pandas as pd
import numpy as np
from rogi import RoughnessIndex, SARI, MODI, RMODI
from scipy.spatial.distance import cosine, pdist, squareform
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

dataset_names = ['CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50',
                 'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50',
                 'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki',
                 'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50',
                 'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki',
                 'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki',
                 'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki',
                 'CHEMBL4616_EC50', 'CHEMBL4792_Ki']
classification_tasks = ["BBBP","hERG","Mutagenicity","oral_bioavailability","HLM_metabolic_stability"]
regression_tasks = ["Caco2","HalfLife","VDss","PAMPA1",'CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50',
                    'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50',
                    'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki',
                    'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50',
                    'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki',
                    'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki',
                    'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki',
                    'CHEMBL4616_EC50', 'CHEMBL4792_Ki']
representations = ["MorganFP"]
discrete = ["MACCSkeys", "RDKFP", "MorganFP", "MorganCount", "phaERGFP", "PubChemFP"]
continuous = ["rdkit_desc", "mordred_desc", "KPGT_Embedding","GEM_Embedding","Uni-Mol_Embedding","K_BERT_Embedding"]

def get_sim_matrix_for_fingerprint(smiles, fingerprints='morgan'):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    if isinstance(fingerprints, str):
        if fingerprints == 'maccs':
            fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
        elif fingerprints == 'morgan':
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, nBits=2048, radius=2) for m in mols]
        else:
            raise ValueError('only "maccs" or "morgan" fingerprints are allowed')
    else:
        fps = fingerprints

    # compute similarity matrix
    sim_matrix = np.zeros(shape=(len(smiles), len(smiles)))
    for i in range(len(smiles)):
        # i+1 because we know the diagonal is zero
        sim_matrix[i, i + 1:] = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
        sim_matrix[i + 1:, i] = sim_matrix[i, i + 1:]
    return sim_matrix

def get_sim_matrix(X=None, scale=False):
    sim_matrix = np.zeros(shape=(len(X), len(X)))
    for i in range(len(X)):
        for j in range(i, len(X)):
            # Calculate cosine similarity
            sim_matrix[i, i] = 1  # Fill the diagonal
            cosine_sim = 1 - cosine(X[i], X[j])
            if scale:
                sim_matrix[i, j] = (cosine_sim + 1)/2
            else:
                sim_matrix[i, j] = cosine_sim  # 1 - cosine distance gives similarity
            sim_matrix[j, i] = sim_matrix[i, j]  # Symmetrical assignment
    return sim_matrix

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
        # i+1 because we know the diagonal is zero
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

print("Calculating SARI using a dynamic threshold, and adding chirality to MorganFP for ROGI and MODI")
# SARI
print('SARI')
SARi = []
for task in dataset_names:
    print('task: ',task)
    roughness = {}
    roughness['task'] = task
    origin_data = pd.read_csv(f"./data/{task}.csv")
    origin_data1 = pd.read_csv(f'./data/origin_data/{task}_MoleculeACE_2024.csv')
    smiless = origin_data1['smiles'].tolist()
    MACCSkeys_similarity = extract_upper_tri(calculate_tanimoto(smiless, "MACCSkeys"))
    for representation in representations:
        print('representation: ', representation)
        if representation in ('MorganFP'):
            data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_MoleculeACE_2024_{representation}.csv")
            smiless = data_with_representation['smiles'].tolist()
            MorganFP_similarity_matrix = calculate_tanimoto(smiless, "morgan")
            MorganFP_similarity = extract_upper_tri(MorganFP_similarity_matrix)
            p = np.mean(MACCSkeys_similarity > 0.6)
            t_morgan = np.percentile(MorganFP_similarity, 100 * (1 - p))
            print(f"MACCS similarity proportion: {p:.2%}")
            print(f"Morgan equivalent threshold: {t_morgan:.4f}")
            sari = SARI(pKi=data_with_representation[task], sim_matrix=MorganFP_similarity_matrix)
            Sari, raw_cont, raw_disc = sari.compute_sari(similarity_threshold=t_morgan)
            roughness[representation] = Sari
            roughness[f'{representation}_raw_cont'] = raw_cont
            roughness[f'{representation}_raw_disc'] = raw_disc
            print(roughness)
        elif representation in ('rdkit_desc', 'mordred_desc'):
            data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_MoleculeACE_2024_{representation}.csv")
            features = data_with_representation.columns.difference(['smiles', task, 'group'])
            X=data_with_representation[features].values
            Descriptors_similarity_matrix = calculate_cosine(X)
            Descriptors_similarity = extract_upper_tri(Descriptors_similarity_matrix)
            p = np.mean(MACCSkeys_similarity > 0.6)
            t_descriptors = np.percentile(Descriptors_similarity, 100 * (1 - p))
            print(f"MACCS similarity proportion: {p:.2%}")
            print(f"Descriptors equivalent threshold: {t_descriptors:.4f}")
            sari = SARI(pKi=data_with_representation[task], sim_matrix=Descriptors_similarity_matrix)
            Sari, raw_cont, raw_disc = sari.compute_sari(similarity_threshold=t_descriptors)
            roughness[representation] = Sari
            roughness[f'{representation}_raw_cont'] = raw_cont
            roughness[f'{representation}_raw_disc'] = raw_disc
            print(roughness)
        else:
            data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_{representation}.csv")
            merged_data = pd.merge(origin_data1, data_with_representation, on=['smiles'], how='inner')
            features = merged_data.columns.difference(['smiles', task, 'group'])
            merged_data.replace('NaN', np.nan, inplace=True)
            data = merged_data.dropna(subset=features).reset_index(drop=True)
            columns_order = ['smiles', task, 'group'] + [col for col in merged_data.columns if col not in ['smiles', task, 'group']]
            data = data[columns_order]
            X = data[features].values
            Embeddings_similarity_matrix = calculate_cosine(X)
            Embeddings_similarity = extract_upper_tri(Embeddings_similarity_matrix)
            p = np.mean(MACCSkeys_similarity > 0.6)
            t_embeddings = np.percentile(Embeddings_similarity, 100 * (1 - p))
            print(f"MACCS similarity proportion: {p:.2%}")
            print(f"Embeddings equivalent threshold: {t_embeddings:.4f}")
            sari = SARI(pKi=data[task], sim_matrix=Embeddings_similarity_matrix)
            Sari, raw_cont, raw_disc = sari.compute_sari(similarity_threshold=t_embeddings)
            roughness[representation] = Sari
            roughness[f'{representation}_raw_cont'] = raw_cont
            roughness[f'{representation}_raw_disc'] = raw_disc
            print(roughness)
    SARi.append(roughness)
df = pd.DataFrame(SARi)
df.to_csv("SARI(Dynamic_Thresholds)-MoleculeACE.csv", index=False)

# ROGI
print('ROGI')
ROGI = []
for task in dataset_names:
    print('task: ',task)
    roughness = {}
    roughness['task'] = task
    origin_data = pd.read_csv(f"./data/{task}.csv")
    origin_data1 = pd.read_csv(f'./data/origin_data/{task}_MoleculeACE_2024.csv')
    for representation in representations:
        print('representation: ', representation)
        if representation in ('MorganCount','phaERGFP'):
            data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_MoleculeACE_2024_{representation}.csv")
            features = data_with_representation.columns.difference(['smiles', task, 'group'])
            ri = RoughnessIndex(Y=data_with_representation[task], X=data_with_representation[features], metric='euclidean')
            roughness[representation] = ri.compute_index()
            print(roughness)
        elif representation in discrete:
            data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_MoleculeACE_2024_{representation}.csv")
            features = data_with_representation.columns.difference(['smiles', task, 'group'])
            smiless = data_with_representation['smiles'].tolist()
            ri = RoughnessIndex(Y=data_with_representation[task], smiles=smiless)
            roughness[representation] = ri.compute_index()
            print(roughness)
        elif representation in ('rdkit_desc', 'mordred_desc'):
            if task=='PAMPA1':
                data_with_representation = pd.read_csv(f"data/processed_data/{representation}/PAMPA_MoleculeACE_2024_{representation}.csv")
                features = data_with_representation.columns.difference(['smiles', 'PAMPA', 'group'])
                ri = RoughnessIndex(Y=data_with_representation['PAMPA'], X=data_with_representation[features], metric='cosine')
                roughness[representation] = ri.compute_index()
            else:
                data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_MoleculeACE_2024_{representation}.csv")
                features = data_with_representation.columns.difference(['smiles', task, 'group'])
                ri = RoughnessIndex(Y=data_with_representation[task], X=data_with_representation[features], metric='cosine')
                roughness[representation] = ri.compute_index()
            print(roughness)
        else:
            data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_{representation}.csv")
            merged_data = pd.merge(origin_data1, data_with_representation, on=['smiles'], how='inner')
            features = merged_data.columns.difference(['smiles', task, 'group'])
            merged_data.replace('NaN', np.nan, inplace=True)
            data = merged_data.dropna(subset=features)
            columns_order = ['smiles', task, 'group'] + [col for col in merged_data.columns if col not in ['smiles', task, 'group']]
            data = data[columns_order]
            ri = RoughnessIndex(Y=data[task], X=data[features], metric='cosine')
            roughness[representation] = ri.compute_index()
            print(roughness)
    ROGI.append(roughness)
df = pd.DataFrame(ROGI)
df.to_csv("ROGI-MoleculeACE.csv", index=False)

print('Recalculating MODI')
# MODI
MODi = []
for task in dataset_names:
    print('task: ',task)
    roughness = {}
    roughness['task'] = task
    origin_data = pd.read_csv(f"./data/{task}.csv")
    origin_data1 = pd.read_csv(f'./data/origin_data/{task}_MoleculeACE_2024.csv')
    for representation in representations:
        print('representation: ', representation)
        if representation in ('MorganCount','phaERGFP'):
            pass
        elif representation in discrete:
            data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_MoleculeACE_2024_{representation}.csv")
            features = data_with_representation.columns.difference(['smiles', task, 'group'])
            smiless = data_with_representation['smiles'].tolist()
            X=data_with_representation[features].values
            if task in classification_tasks:
                roughness[representation] = MODI(Dx=1-calculate_tanimoto(smiless, fp_type='morgan'), Y=data_with_representation[task].tolist())
            else:
                roughness[representation] = RMODI(Dx=1-calculate_tanimoto(smiless, fp_type='morgan'), Y=data_with_representation[task].tolist())
            print(roughness)
        elif representation in ('rdkit_desc', 'mordred_desc'):
            data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_MoleculeACE_2024_{representation}.csv")
            features = data_with_representation.columns.difference(['smiles', task, 'group'])
            X=data_with_representation[features].values
            if task in classification_tasks:
                roughness[representation] = MODI(Dx=calculate_cosine_distance(X), Y=data_with_representation[task].tolist())
            else:
                roughness[representation] = RMODI(Dx=calculate_cosine_distance(X), Y=data_with_representation[task].tolist())
            print(roughness)
        else:
            data_with_representation = pd.read_csv(f"data/processed_data/{representation}/{task}_{representation}.csv")
            merged_data = pd.merge(origin_data1, data_with_representation, on=['smiles'], how='inner')
            features = merged_data.columns.difference(['smiles', task, 'group'])
            merged_data.replace('NaN', np.nan, inplace=True)
            data = merged_data.dropna(subset=features)
            columns_order = ['smiles', task, 'group'] + [col for col in merged_data.columns if col not in ['smiles', task, 'group']]
            data = data[columns_order]
            X = data[features].values
            if task in classification_tasks:
                roughness[representation] = MODI(Dx=calculate_cosine_distance(X), Y=data[task].tolist())
            else:
                roughness[representation] = RMODI(Dx=calculate_cosine_distance(X), Y=data[task].tolist())
            print(roughness)
    MODi.append(roughness)
df = pd.DataFrame(MODi)
df.to_csv("MODI-MoleculeACE.csv", index=False)