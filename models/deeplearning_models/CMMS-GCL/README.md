# CMMS-GCL

CMMS-GCL (Cross-Modality Metabolic Stability Prediction with Graph Contrastive Learning) combines graph convolutional networks and SMILES sequence representations with contrastive learning.

## Data Representation

CMMS-GCL uses dual molecular representations:

- **Graph representation**: Molecular graph with PyG (PyTorch Geometric)
- **Sequence representation**: SMILES sequences via Smi2Vec
- **Features**: Character-level embeddings for SMILES

## How to Run

1. **Install dependencies**:
   ```bash
   pip install torch scikit-learn pandas numpy rdkit networkx pyg
   ```

2. **Prepare data**:
   - Place training data in the data directory
   - Format: CSV with SMILES and labels

3. **Run training**:
   ```python
   from cmms_gcl import train

   train(dataset='HLM',
         epochs=500,
         batch_size=256,
         lr=0.0005)
   ```

## Model Architecture

- **Graph encoder**: GCN (Graph Convolutional Network)
- **Sequence encoder**: Bidirectional GRU
- **Fusion**: Concatenation of graph and sequence embeddings
- **Contrastive learning**: Inter-view graph contrastive learning
