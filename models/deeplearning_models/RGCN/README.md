# RGCN

RGCN (Relational Graph Convolutional Network) is a GNN architecture that handles edge features by transforming heterogeneous graphs into multiple homogeneous graphs.

## Data Representation

- **Input**: SMILES strings
- **Graph format**: DGL graphs
- **Node features**: Atomic properties (40 dimensions with chirality, 37 without)
- **Edge features**: Bond properties with relation types
- **Data files**: `{dataset_name}.bin` and `{dataset_name}_group.csv` in `data/graph_data/`

## How to Run

1. **Build graph dataset**:
   ```bash
   python build_dataset.py
   ```

2. **Train model**:
   ```python
   from train_rgcn import train_RGCN

   train_RGCN(
       times=3,
       task_name='BBBP',
       data_name='BBBP',
       split_method='random',
       classification=True
   )
   ```

   Key parameters:
   - `times`: Number of training runs
   - `task_name`: Task name
   - `data_name`: Dataset name
   - `split_method`: 'random', 'scaffold', or 'Perimeter'
   - `classification`: True for classification, False for regression
