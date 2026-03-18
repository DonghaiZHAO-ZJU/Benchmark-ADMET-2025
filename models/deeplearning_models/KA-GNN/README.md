# KA-GNN

KA-GNN (Kolmogorov-Arnold Graph Neural Network) integrates Kolmogorov-Arnold Networks (KAN) into GNN architectures.

## Data Representation

- **Input**: SMILES strings
- **Node features**: Atom properties (92 dimensions)
- **Edge features**: Bond properties (21 dimensions)
- **Additional**: Non-covalent interactions (5 Å spatial cutoff)

## How to Run

Run training:
```bash
cd KA-GNN
bash run.sh
```

Or run directly:
```bash
cd KA-GNN
python main.py
```

Configure dataset in `./KA-GNN/config/c_path.yaml`:
```yaml
select_dataset: "BBBP"
```
