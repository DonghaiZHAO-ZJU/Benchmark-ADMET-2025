# AutoGluon

AutoGluon is an AutoML framework that automates model training, hyperparameter tuning, and ensemble for tabular data.

## Data Representation

- **Input**: CSV files with SMILES and labels
- **Features**: Molecular fingerprints (Morgan, MACCS, RDKit) and molecular descriptors
- **Data location**: `./data/`

## How to Run

Run training:
```bash
python autogln.py --task BBBP --split_method random --split_seed 2024
```

Key parameters:
- `--task`: Name of the task
- `--split_method`: `random` / `scaffold` / `Perimeter`
- `--split_seed`: Split seed, e.g. `2024`
- `--eval_metric`: Evaluation metric
- `--time_limit`: Training time limit in seconds (default 3600)
