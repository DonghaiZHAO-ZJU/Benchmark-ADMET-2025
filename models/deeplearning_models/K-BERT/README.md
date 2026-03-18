# K-BERT

K-BERT (Knowledge-based BERT) is a BERT-based molecular representation model that learns molecular features from SMILES sequences.

## Data Representation

- **Input**: SMILES strings
- **Pretraining**: Atomic feature prediction, molecular feature prediction, contrastive learning
- **Supports**: Non-canonical SMILES

## How to Run

Run training:
```bash
python train.py --data_path path/to/data --model_path path/to/pretrained_model
```

Key parameters:
- `--data_path`: Path to training data
- `--model_path`: Path to pretrained model
- `--task`: Task type (classification/regression)
- `--epochs`: Number of training epochs
