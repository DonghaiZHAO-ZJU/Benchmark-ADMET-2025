# Classical ML Models

This directory contains traditional machine learning models for molecular property prediction using molecular fingerprints and descriptors.

## Data Representation

- **Input**: CSV files with SMILES and labels
- **Features**: Molecular fingerprints (Morgan FP, MACCS keys, RDKit 2D descriptors)
- **Data location**: `./data/processed_data1/`

## How to Run

Each model can be run independently:

**XGBoost**:
```bash
python XGBoost.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
```

**LightGBM**:
```bash
python LightGBM.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
```

**CatBoost**:
```bash
python CatBoost.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
```

**SVM**:
```bash
python SVM.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
```

**KNN**:
```bash
python KNN.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
```

**DNN**:
```bash
python DNN_script.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
```

Key parameters:
- `--data_name`: Dataset name
- `--task_name`: Task name
- `--task_type`: 'classification' or 'regression'
- `--fp_name`: Fingerprint name (MorganFP, MACCS, RDKFP, etc.)
- `--seed`: Random seed
