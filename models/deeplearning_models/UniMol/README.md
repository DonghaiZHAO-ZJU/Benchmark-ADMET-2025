# Uni-Mol

Uni-Mol is a pretrained molecular representation model based on 3D molecular structures.

## Data Representation

Uni-Mol uses 3D molecular structures:

- **Input format**: SMILES strings (Uni-Mol computes 3D conformations internally)
- **3D coordinates**: Generated from SMILES using RDKit
- **Data files**: CSV files with SMILES and labels

## How to Run

1. **Install Uni-Mol**:
   ```bash
   pip install unimol
   ```

2. **Run training**:
   ```bash
   python unimol.py --task BBBP --split_method random --seed 2024 --data_type classification --metric roc_auc
   ```

   Key parameters:
   - `--task`: Task/dataset name
   - `--split_method`: 'random', 'scaffold', 'Perimeter', 'Maximum_Dissimilarity'
   - `--seed`: Random seed
   - `--data_type`: 'classification' or 'regression'
   - `--metric`: 'roc_auc' (classification), 'r2' (regression)

## Model Architecture

- **Pretrained model**: Based on 3D molecular structure
- **Position encoding**: Rotation and translation invariant spatial encoding
- **Attention**: Pairwise representation with query-key product
- **Tasks**: 3D position denoising, distance prediction

## References

- Zhou, G. et al. (2022). Uni-Mol: A Universal 3D Molecular Representation Learning Framework.
