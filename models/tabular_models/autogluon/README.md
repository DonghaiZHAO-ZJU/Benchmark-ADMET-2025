# AutoGluon

AutoGluon is an AutoML framework that automates model training, hyperparameter tuning, and ensemble for tabular data.

## Data Representation

- **Input**: CSV files with SMILES and labels
- **Features**: Molecular fingerprints (Morgan, MACCS, RDKit) and molecular descriptors
- **Data location**: `./data/`

## How to Run

Run training:
```bash
python autogln.py
```

Key parameters (in code):
- `task_name`: Name of the task
- `task_type`: 'classification' or 'regression'
- `time_limit`: Training time limit in seconds
