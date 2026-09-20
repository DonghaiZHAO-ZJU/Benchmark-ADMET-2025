# K-BERT

K-BERT (Knowledge-based BERT) is a BERT-based molecular representation model that learns molecular features from SMILES sequences.

## Data Representation

- **Input**: SMILES strings
- **Pretraining**: Atomic feature prediction, molecular feature prediction, contrastive learning
- **Supports**: Non-canonical SMILES

## How to Run

Run training:
```bash
bash sh/K_BERT.sh
```

`sh/K_BERT.sh` loops over 5 seeds (2024-2064) and 4 split methods, calling:

```bash
python practice.py --seed $seed --split_method $split_method --scaler StandardScaler
```

For classification tasks use `sh/K_BERT_classification.sh` instead (no scaler).
To build the input data first, run `bash sh/K_BERT_data.sh` (calls `build_dataset_for_tasks.py`).

Key parameters:
- `--seed`: Random seed, one of 2024, 2034, 2044, 2054, 2064
- `--split_method`: 'random', 'scaffold', 'Perimeter', 'Maximum_Dissimilarity' or 'MoleculeACE'
- `--scaler`: 'StandardScaler', 'PowerTransformer' or 'RobustScaler'
