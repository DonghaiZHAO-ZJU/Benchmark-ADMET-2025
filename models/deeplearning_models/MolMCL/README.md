# MolMCL

MolMCL is a prompt-based multi-channel learning framework that integrates molecular contrastive learning, scaffold contrastive learning, and context prediction for molecular property prediction.

## Data Representation

MolMCL uses molecular graphs:

- **Input format**: SMILES strings
- **Graph encoder**: GINE or GPS
- **Data files**: CSV files in `./data/`
- **Config**: YAML files in `./config/`

## How to Run

1. **Install MolMCL**:
   ```bash
   pip install -e .
   ```

2. **Run training**:
   ```bash
   bash finetune_example.sh
   ```

   Or run directly:
   ```bash
   python scripts/finetune.py \
       --config ./config/ADMET_example.yaml \
       --split_type random_2024 \
       --data_name BBBP
   ```

   Key parameters:
   - `--config`: Configuration file path
   - `--split_type`: Split type and seed (e.g., 'random_2024')
   - `--data_name`: Dataset name

## Model Architecture

- **Encoder**: GINE or GPS (Graph Pooling + Self-attention)
- **Channels**: 3 prompt-tagged channels
  1. Molecular contrastive learning
  2. Scaffold contrastive learning
  3. Context prediction
- **Prompt-weighted pooling**: Selects prompt weights based on Roughness Index