# AttentiveFP

AttentiveFP is a Graph Neural Network (GNN) architecture that uses graph attention mechanisms to enhance molecular representation capabilities.

## Data Representation

AttentiveFP represents molecules as graph data structures using DGL (Deep Graph Library):

- **Node features**: Atomic properties (40 dimensions including atom type, degree, formal charge, etc.)
- **Edge features**: Bond properties (bond type, conjugated, in ring, etc.)
- **Graph format**: Binary `.bin` files stored in `data/Attentivefp_graph_data/`
- **Data format**: `{dataset_name}.bin` and `{dataset_name}_group.csv`

## How to Run

1. **Prepare graph data**:
   ```bash
   python build_graph_dataset.py
   ```

2. **Train the model**:
   ```bash
   python AttentiveFP_singletask_model.py
   ```

   Key parameters:
   - `times`: Number of training runs
   - `task_name`: Name of the task
   - `data_name`: Name of the dataset
   - `classification`: Set to True/False for classification/regression

3. **Example** (from `practice.py`):
   ```python
   from AttentiveFP_singletask_model import AttentiveFP_model

   AttentiveFP_model(
       times=3,
       task_name='BBBP',
       data_name='BBBP',
       classification=True
   )
   ```

## Model Architecture

- **Input**: Molecular graph (node/edge features)
- **Layers**: 6 graph attention layers
- **Readout**: 2 timesteps of attention-based pooling
- **Hidden size**: 200
- **Output**: Task-specific prediction (classification/regression)
