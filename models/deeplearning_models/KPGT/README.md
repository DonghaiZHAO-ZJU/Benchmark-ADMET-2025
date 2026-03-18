# KPGT

KPGT (Knowledge-Pretrained Graph Transformer) uses a Line Graph Transformer (LiGhT) architecture with pretrained molecular representations.

## Data Representation

- **Input**: SMILES strings
- **Graph type**: Line graph (nodes = atom pairs, edges = bonds)
- **Features**: Path encoding and distance encoding
- **Data location**: `./datasets/`

## How to Run

Run training:
```bash
cd scripts
python finetune_new.py --config base \
    --model_path ../models/pretrained/base/base.pth \
    --dataset BBBP \
    --data_path ../datasets/ \
    --dataset_type classification \
    --metric rocauc \
    --split "random_2024"
```

Key parameters:
- `--config`: Model configuration ('base')
- `--model_path`: Path to pretrained model
- `--dataset`: Dataset/task name
- `--dataset_type`: 'classification' or 'regression'
- `--metric`: Evaluation metrics
- `--split`: Data split name
