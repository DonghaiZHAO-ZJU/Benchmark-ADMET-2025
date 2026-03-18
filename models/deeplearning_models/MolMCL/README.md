# MolMCL

MolMCL is a prompt-based multi-channel learning framework that integrates molecular contrastive learning, scaffold contrastive learning, and context prediction.

## Data Representation

- **Input**: SMILES strings
- **Graph encoder**: GINE or GPS
- **Data format**: CSV files in `./data/`

## How to Run

Run training:
```bash
bash finetune_example.sh
```

Or run with Python:
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
