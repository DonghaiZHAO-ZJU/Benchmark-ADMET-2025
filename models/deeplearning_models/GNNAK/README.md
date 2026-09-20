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
python admet.py --cfg configs/admet.yaml
```

Key parameters (in config file):
- `dataset`: Dataset name (e.g. `admet`)
- `subgraph.hops`: Number of hops for subgraph extraction
- `subgraph.online`: Whether subgraph sampling is done online
- `train.runs` / `train.epochs` / `train.patience`: Training loop settings
