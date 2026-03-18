# Uni-Mol

Uni-Mol is a pretrained molecular representation model based on 3D molecular structures.

## Data Representation

- **Input**: SMILES strings (Uni-Mol computes 3D conformations internally)
- **3D coordinates**: Generated from SMILES using RDKit
- **Data format**: CSV files with SMILES and labels

## How to Run

Run training:
```bash
python unimol.py --task BBBP --split_method random --seed 2024 --data_type classification --metric roc_auc
```

Key parameters:
- `--task`: Task/dataset name
- `--split_method`: 'random', 'scaffold', 'Perimeter', 'Maximum_Dissimilarity'
- `--seed`: Random seed
- `--data_type`: 'classification' or 'regression'
- `--metric`: 'roc_auc' (classification), 'r2' (regression)
