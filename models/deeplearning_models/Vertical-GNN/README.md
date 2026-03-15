# Vertical-GNN

Vertical-GNN combines Graph Transformer and GIN (Graph Isomorphism Network) for molecular property prediction, specifically designed for oral bioavailability prediction.

## Data Representation

Vertical-GNN uses molecular graphs:

- **Input format**: SMILES strings converted to graphs
- **Graph features**: Node (atom) and edge (bond) features
- **Additional features**: Molecular fingerprints and descriptors

## How to Run

1. **Prepare data**:
   - Place graph data in `./data/graph_data/`
   - Place fingerprints in `./data/oral_avail_fingerprints/`
   - Place molecular descriptors in `./data/oral_mol_desc/`

2. **Install dependencies**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Run training**:
   Use the Jupyter notebooks in `./notebooks/`:
   ```bash
   jupyter notebook notebooks/transfer_learning_model.ipynb
   ```

## Model Architecture

- **Encoders**: Graph Transformer + GIN
- **Fusion**: Concatenation of features from both encoders
- **Classifier/Regressor**: MLP head for final prediction

## References

- Wang, Y. et al. (2023). Evaluating the Use of GNNs and Transfer Learning for Oral Bioavailability Prediction.
