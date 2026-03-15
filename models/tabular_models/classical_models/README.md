# Classical ML Models

This directory contains traditional machine learning models for molecular property prediction using molecular fingerprints and descriptors.

## Data Representation

Classical models use molecular fingerprints:

- **Input format**: CSV files with SMILES and labels
- **Features**: Various molecular fingerprints
  - Morgan FP (ECFP)
  - MACCS keys
  - RDKit 2D descriptors
- **Data location**: `./data/processed_data1/`

## How to Run

Each model can be run independently:

1. **XGBoost**:
   ```bash
   python XGBoost.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
   ```

2. **LightGBM**:
   ```bash
   python LightGBM.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
   ```

3. **CatBoost**:
   ```bash
   python CatBoost.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
   ```

4. **Random Forest** (if available):
   ```bash
   python RandomForest.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
   ```

5. **SVM**:
   ```bash
   python SVM.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
   ```

6. **KNN**:
   ```bash
   python KNN.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
   ```

7. **DNN**:
   ```bash
   python DNN_script.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
   ```

Key parameters:
- `--data_name`: Dataset name
- `--task_name`: Task name
- `--task_type`: 'classification' or 'regression'
- `--fp_name`: Fingerprint name (MorganFP, MACCS, RDKFP, etc.)
- `--seed`: Random seed

## Model Architectures

| Model | Description |
|-------|-------------|
| **XGBoost** | Gradient boosting with regularization |
| **LightGBM** | Gradient boosting with GOSS sampling |
| **CatBoost** | Gradient boosting with ordered boosting |
| **Random Forest** | Ensemble of decision trees |
| **SVM** | Support Vector Machine with kernel |
| **KNN** | K-Nearest Neighbors |
| **DNN** | Deep Neural Network (MLP) |

## Hyperparameter Optimization

Models use Hyperopt for hyperparameter tuning:
- Search space defined in each script
- Bayesian optimization (TPE)

## References

- Chen, T. et al. (2016). XGBoost: A Scalable Tree Boosting System.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
- Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features.
