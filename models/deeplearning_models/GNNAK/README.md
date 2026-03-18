# GNNAK

GNNAK (GNN As Kernel) is a framework that extends local aggregation in MPNNs from a star pattern to a general subgraph pattern, making it more expressive than 1-WL and 2-WL tests.

## Data Representation

GNNAK uses molecular graphs with subgraph extraction:

- **Input format**: SMILES strings converted to DGL graphs
- **Subgraph extraction**: Random walk-based subgraph sampling
- **Data location**: `./data/admet/`
- **Config files**: YAML configs in `./train/configs/`

## How to Run

1. **Prepare data**: Place processed data in `./data/admet/`

2. **Run training**:
   ```bash
   cd train
   python admet.py --cfg configs/gnnak_admet.yaml
   ```

   Key parameters (in config file):
   - `dataset_name`: Dataset name
   - `dataset_type`: 'classification' or 'regression'
   - `subgraph.hops`: Number of hops for subgraph extraction
   - `subgraph.walk_length`: Random walk length

## Model Architecture

- **Kernel GNN**: Graph Transformer or GINE as base encoder
- **Subgraph pattern**: Induces subgraph centered on each node
- **Variants**: GNNAK, GNNAK+, GNNAK+-S
- **Aggregation**: Subgraph encoding for each node
