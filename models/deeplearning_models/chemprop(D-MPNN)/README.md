# Chemprop (D-MPNN)

Chemprop is a message passing neural network (MPNN) based package for molecular property prediction. The D-MPNN treats directed edges as nodes to facilitate message passing.

## Data Representation

- **Input**: SMILES strings
- **Data format**: CSV files with SMILES and labels
- **Features**: Automatically computes molecular fingerprints and descriptors
- **Data files**: `{task}_{split_method}_{seed}_training.csv`, `_valid.csv`, `_test.csv`

## How to Run

Run training:
```bash
bash run.sh
```

Or run with chemprop command:
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
