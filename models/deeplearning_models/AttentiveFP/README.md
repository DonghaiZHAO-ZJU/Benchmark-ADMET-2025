# AttentiveFP

AttentiveFP is a Graph Neural Network (GNN) architecture that uses graph attention mechanisms to enhance molecular representation capabilities.

## Data Representation

- **Input**: SMILES strings
- **Graph format**: DGL graphs
- **Node features**: Atomic properties (40 dimensions)
- **Edge features**: Bond properties
- **Data files**: `{dataset_name}.bin` and `{dataset_name}_group.csv` in `data/Attentivefp_graph_data/`

## How to Run

1. **Build graph dataset**:
   ```bash
   python build_graph_dataset.py
   ```

2. **Train model**:
   ```python
   from AttentiveFP_singletask_model import AttentiveFP_model

   AttentiveFP_model(
       times=3,
       task_name='BBBP',
       data_name='BBBP',
       classification=True
   )
   ```

   Key parameters:
   - `times`: Number of training runs
   - `task_name`: Task name
   - `data_name`: Dataset name
   - `classification`: True for classification, False for regression
