# RGCN

RGCN (Relational Graph Convolutional Network) is a GNN architecture that handles edge features by transforming heterogeneous graphs into multiple homogeneous graphs.

## Data Representation

RGCN uses DGL graph representation:

- **Node features**: Atomic properties (40 dimensions with chirality, 37 without)
- **Edge features**: Bond properties with relation types
- **Graph format**: Binary `.bin` files in `data/graph_data/`
- **Data files**: `{dataset_name}.bin` and `{dataset_name}_group.csv`

## How to Run

1. **Build graph dataset**:
   ```bash
   python build_dataset.py
   ```

2. **Train the model**:
   ```bash
   python train_rgcn.py
   ```

   Key parameters in `train_rgcn.py`:
   ```python
   train_RGCN(
       times=3,
       task_name='BBBP',
       data_name='BBBP',
       split_method='random',
       classification=True
   )
   ```

   Parameters:
   - `times`: Number of training runs
   - `task_name`: Task name
   - `data_name`: Dataset name
   - `split_method`: 'random', 'scaffold', or 'random'
   - `classification`: True/False

## Model Architecture

- **Input**: Molecular graph with relation edges
- **RGCN layers**: 2 hidden layers (64, 64)
- **FFN**: 1 hidden layer (64)
- **Dropout**: 0.2
- **Optimizer**: Adam with learning rate 3e-3
- **Epochs**: 500 with early stopping (patience 50)
