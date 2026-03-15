# KA-GNN

KA-GNN (Kolmogorov-Arnold Graph Neural Network) integrates Kolmogorov-Arnold Networks (KAN) into GNN architectures for molecular property prediction.

## Data Representation

KA-GNN uses molecular graphs with enhanced spatial features:

- **Input format**: SMILES strings
- **Node features**: Atom properties (92 dimensions)
- **Edge features**: Bond properties (21 dimensions)
- **Additional**: Non-covalent interactions (5 Å spatial cutoff)

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure dataset**:
   Modify `./KA-GNN/config/c_path.yaml`:
   ```yaml
   select_dataset: "BBBP"
   ```

3. **Run training**:
   ```bash
   cd KA-GNN
   python main.py
   ```

   Or use shell scripts:
   ```bash
   bash run.sh
   ```

## Model Architecture

- **Variants**: KA-GCN and KA-GAT
- **KAN integration**: Node embedding, message passing, and readout modules
- **Activation**: Fourier series-based univariate functions
- **Layers**: 3 Fourier GNN layers
- **Readout**: 2 KAN Linear layers

## References

- Li, L. et al. (2025). Kolmogorov-Arnold Graph Neural Networks for Molecular Property Prediction. Nature Machine Intelligence.
