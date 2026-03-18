# K-BERT

K-BERT (Knowledge-based BERT) is a BERT-based molecular representation model that learns molecular features from SMILES sequences.

## Data Representation

K-BERT uses SMILES strings:

- **Input format**: SMILES sequences
- **Pretraining**: Atomic feature prediction, molecular feature prediction, contrastive learning
- **Supports**: Non-canonical SMILES

## How to Run

1. **Install dependencies**:
   ```bash
   pip install rdkit pytorch sklearn xgboost
   ```

2. **Download pretrained model and datasets**:
   - Download from: https://pan.baidu.com/s/1yzhHwhELuJG-3lxlrVtRPA (code: WZXX)

3. **Run training**:
   Use the scripts in the repository:
   ```bash
   python train.py --data_path path/to/data --model_path path/to/pretrained_model
   ```

## Model Architecture

- **Architecture**: Transformer encoder
- **Pretraining tasks**:
  1. Atom feature prediction (degree, aromaticity, hydrogens, chirality)
  2. Molecular feature prediction (MACCS fingerprints)
  3. Contrastive learning (same molecule, different SMILES)