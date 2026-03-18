# CMMS-GCL

CMMS-GCL combines graph convolutional networks and SMILES sequence representations with contrastive learning for metabolic stability prediction.

## Data Representation

- **Input**: SMILES strings
- **Graph representation**: PyG graphs
- **Sequence representation**: Smi2Vec character embeddings
- **Data format**: CSV with SMILES and labels

## How to Run

Run training:
```python
from cmms_gcl import train

train(dataset='HLM',
      epochs=500,
      batch_size=256,
      lr=0.0005)
```

Key parameters:
- `dataset`: Dataset name
- `epochs`: Number of training epochs
- `batch_size`: Batch size
- `lr`: Learning rate
- `classification`: True/False
