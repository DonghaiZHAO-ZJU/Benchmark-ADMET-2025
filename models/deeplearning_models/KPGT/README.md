# KPGT

KPGT (Knowledge-Pretrained Graph Transformer) uses a Line Graph Transformer (LiGhT) architecture with pretrained molecular representations.

## Data Representation

KPGT uses molecular line graphs:

- **Input format**: SMILES strings
- **Graph type**: Line graph (nodes = atom pairs, edges = bonds)
- **Features**: Path encoding and distance encoding
- **Data location**: `./datasets/`

## How to Run

1. **Download pretrained model**:
   - Place pretrained weights in `./models/pretrained/base/base.pth`

2. **Run training**:
   ```bash
   cd scripts
   bash run.sh
   ```

   Or run directly:
   ```bash
   python finetune_new.py --config base \
       --model_path ../models/pretrained/base/base.pth \
       --dataset BBBP \
       --data_path ../datasets/ \
       --dataset_type classification \
       --metric rocauc prauc acc \
       --split "random_2024" \
       --lr 3e-5
   ```

   Key parameters:
   - `--config`: Model configuration ('base')
   - `--model_path`: Path to pretrained model
   - `--dataset`: Dataset/task name
   - `--dataset_type`: 'classification' or 'regression'
   - `--metric`: Evaluation metrics
   - `--split`: Data split name

## Model Architecture

- **Architecture**: Line Graph Transformer (LiGhT)
- **Pretraining**: Masked node prediction, RDKit descriptors, ECFP fingerprints
- **Global nodes**: 2 global nodes initialized with molecular descriptors