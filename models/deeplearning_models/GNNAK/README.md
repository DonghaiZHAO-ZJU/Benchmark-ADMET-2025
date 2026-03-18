# GNNAK

GNNAK (GNN As Kernel) is a framework that extends local aggregation in MPNNs from a star pattern to a general subgraph pattern.

## Data Representation

- **Input**: SMILES strings converted to DGL graphs
- **Subgraph extraction**: Random walk-based subgraph sampling
- **Data location**: `./data/admet/`

## How to Run

Run training:
```bash
cd train
python admet.py --cfg configs/gnnak_admet.yaml
```

Key parameters (in config file):
- `dataset_name`: Dataset name
- `dataset_type`: 'classification' or 'regression'
- `subgraph.hops`: Number of hops for subgraph extraction
- `subgraph.walk_length`: Random walk length
