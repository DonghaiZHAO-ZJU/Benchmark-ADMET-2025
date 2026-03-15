# AutoGluon

AutoGluon is an AutoML framework that automates data preprocessing, model training, hyperparameter tuning, and ensemble for tabular data.

## Data Representation

AutoGluon uses molecular fingerprints and descriptors:

- **Input format**: CSV files with SMILES and labels
- **Features**: Molecular fingerprints (Morgan, MACCS, RDKit) and molecular descriptors
- **Data location**: `./data/`
- **Feature generation**: Uses RDKit for molecular descriptors

## How to Run

1. **Install dependencies**:
   ```bash
   pip install autogluon rdkit pandas numpy
   ```

2. **Prepare data**:
   - Place processed data in `./data/`
   - Format: CSV with SMILES, labels, and features

3. **Run training**:
   ```bash
   python autogln.py
   ```

   Key parameters (in code):
   - `task_name`: Name of the task
   - `task_type`: 'classification' or 'regression'
   - `time_limit`: Training time limit in seconds

## Model Architecture

- **AutoML**: Automated model selection and hyperparameter tuning
- **Ensemble**: Multiple models combined for better performance
- **Features**: Tabular features from molecular fingerprints
- **Time limit**: Configurable training time (default: 3600s)

## References

- Erickson, N. et al. (2020). AutoGluon-Tabular: Robust Tabular Classification and Regression.
