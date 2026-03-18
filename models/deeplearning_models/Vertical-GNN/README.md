# Vertical-GNN

Vertical-GNN combines Graph Transformer and GIN (Graph Isomorphism Network) for molecular property prediction.

## Data Representation

- **Input**: SMILES strings
- **Graph features**: Node (atom) and edge (bond) features
- **Additional features**: Molecular fingerprints and descriptors
- **Data location**: `./data/`

## How to Run

Run training:
```bash
bash run.sh
```

Or use the Jupyter notebooks in `./notebooks/`:
```bash
jupyter notebook notebooks/transfer_learning_model.ipynb
```

Key parameters (in notebooks):
- `task_name`: Task name
- `split_method`: 'random', 'scaffold', etc.
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate
