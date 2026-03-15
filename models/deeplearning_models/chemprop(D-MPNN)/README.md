# Chemprop (D-MPNN)

Chemprop is a message passing neural network (MPNN) based package for molecular property prediction. The D-MPNN (Directed Message Passing Neural Network) treats directed edges as nodes to facilitate message passing.

## Data Representation

Chemprop uses SMILES strings as input:

- **Input format**: CSV files with SMILES and labels
- **Features**: Automatically computes molecular fingerprints and descriptors
- **Data files**: `{task}_{split_method}_{seed}_training.csv`, `_valid.csv`, `_test.csv`

## How to Run

1. **Install chemprop** (if not already installed):
   ```bash
   cd /path/to/chemprop
   pip install -e .
   ```

2. **Run training**:
   ```bash
   bash run.sh
   ```

   Or run directly with chemprop commands:
   ```bash
   chemprop_train --data_path path/to/train.csv \
                  --separate_val_path path/to/val.csv \
                  --separate_test_path path/to/test.csv \
                  --dataset_type classification \
                  --save_dir path/to/results \
                  --metric auc \
                  --seed 2024 \
                  --num_folds 5
   ```

   Key parameters:
   - `--dataset_type`: 'classification' or 'regression'
   - `--metric`: 'auc' (classification), 'r2' (regression)
   - `--num_folds`: Number of cross-validation folds
   - `--seed`: Random seed

## Model Architecture

- **Message passing**: Directed Message Passing Neural Network
- **Hidden size**: 300
- **Layers**: 3 MPNN layers
- **FFN**: 2 hidden layers (300)
- **Aggregation**: Readout function for graph-level prediction

## References

- Heid, E. et al. (2023). Chemprop: A Machine Learning Package for Chemical Property Prediction.
