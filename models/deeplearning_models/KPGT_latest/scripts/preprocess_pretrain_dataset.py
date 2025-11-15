import sys
sys.path.append("..")

import numpy as np
from multiprocessing import Pool
from rdkit import Chem
from scipy import sparse as sp
import argparse 

from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args

def preprocess_dataset(args):
    with open(f"{args.data_path}/smiles.smi", 'r') as f:
            lines = f.readlines()
            smiless = [line.strip('\n') for line in lines]

    # print('extracting fingerprints', flush=True)
    # FP_list = []
    L = len(smiless)
    # for smiles in tqdm(smiless, total=L):
    #     mol = Chem.MolFromSmiles(smiles)
    #     FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    # FP_arr = np.array(FP_list)
    # FP_sp_mat = sp.csc_matrix(FP_arr)
    # print('saving fingerprints', flush=True)
    # sp.save_npz(f"{args.data_path}/rdkfp1-7_512.npz", FP_sp_mat)

    # print('extracting molecular descriptors', flush=True)
    # generator = RDKit2DNormalized()
    # features_map = Pool(args.n_jobs).imap(generator.process, smiless)
    # arr = np.array(list(features_map))
    # np.savez_compressed(f"{args.data_path}/molecular_descriptors.npz",md=arr[:,1:])

    print('Extracting molecular descriptors', flush=True)
    generator = RDKit2DNormalized()

    # 使用 tqdm 包装 features_map 以显示进度条
    with Pool(args.n_jobs) as pool:
        features_map = pool.imap(generator.process, smiless)
        arr = np.array(list(tqdm(features_map, total=L, desc='Processing Descriptors')))
    np.savez_compressed(f"{args.data_path}/molecular_descriptors.npz", md=arr[:, 1:])

if __name__ == '__main__':
    args = parse_args()
    preprocess_dataset(args)