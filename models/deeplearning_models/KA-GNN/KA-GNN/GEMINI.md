# KA-GNN (Knowledge-Augmented Graph Neural Network)

A Graph Neural Network project for molecular property prediction (ADMET modeling), implementing various GNN architectures including KA-GNN, KAN-GNN, and MLP-GNN.

## Project Overview

- **Purpose:** Prediction of molecular properties (classification and regression) using advanced GNN architectures.
- **Key Technologies:**
  - **Frameworks:** [PyTorch](https://pytorch.org/), [DGL (Deep Graph Library)](https://www.dgl.ai/)
  - **Cheminformatics:** [RDKit](https://rdkit.org/)
  - **Data Analysis:** [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/)
  - **Configuration:** [YAML](https://yaml.org/)
- **Architecture:**
  - **Models:** Located in `model/` (e.g., `ka_gnn.py`, `kan_sage.py`, `mlp_sage.py`).
  - **Encoders:** Supports various atom (e.g., `cgcnn`, `basic`) and bond (e.g., `dim_14`) encoding methods.
  - **Splitting Methods:** Supports `random`, `scaffold`, and `Perimeter` splitting for evaluation robustness.

## Directory Structure

- `main.py`: The primary entry point for training and evaluation.
- `config/`: Contains `c_path.yaml` for model selection and hyperparameter configuration.
- `data/`:
  - `origin_data/`: Raw CSV files containing SMILES and labels.
  - `processed_data/`: Cached DGL graphs and processed datasets (suffix `.pth`).
- `model/`: Implementation of different GNN layers and models.
- `model_weights/`: Saved PyTorch model weights (`.pth`).
- `predictions/`: CSV files with test set predictions and labels.
- `result/`: Final evaluation metrics (AUC, Accuracy, R2, MSE, etc.).
- `utils/`: Utility scripts for data splitting and graph construction.
- `logs/`: Execution logs.

## Building and Running

### Environment Setup
The project typically runs in a Conda environment.
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ka-gnn
```

### Running a Single Task
Use `main.py` with the following required arguments:
```bash
python main.py --select_dataset <dataset> --loss_sclect <bce|l2|l1> --split_method <random|scaffold|Perimeter> --seed <seed>
```
Example:
```bash
python main.py --select_dataset bace --loss_sclect bce --split_method scaffold --seed 2024
```

### Batch Execution
The project provides several shell scripts (`run.sh`, `run1.sh` through `run5.sh`) to automate experiments across multiple datasets, split methods, and seeds.

### Configuration
Global settings can be modified in `config/c_path.yaml`:
- `model_select`: Choose between `ka_gnn`, `mlp_sage`, `kan_sage`, etc.
- `NUM_EPOCHS`, `LR`, `batch_size`: Training hyperparameters.
- `encoder_atom`, `encoder_bond`: Feature encoding selections.

## Development Conventions

- **Data Handling:** SMILES are converted to DGL graphs using `utils/graph_path.py`. Caching is used to speed up repeated runs.
- **Model Evaluation:** 
  - **Classification:** Evaluated using ROC-AUC, Accuracy, and PR-AUC.
  - **Regression:** Evaluated using R², RMSE, and MAE.
- **Weights Management:** Weights are saved per dataset, split, and seed in `model_weights/`. Incomplete weight files should be cleaned before re-running tasks.
- **Logging:** Uses `logzero` for structured logging during training.
