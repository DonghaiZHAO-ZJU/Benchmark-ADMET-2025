# TabPFNv2

TabPFNv2 is a pretrained Transformer-based foundation model for tabular data.

## Data Representation

- **Input**: CSV files with SMILES and labels
- **Features**: Molecular fingerprints (Morgan, MACCS, etc.)
- **Data location**: `./data/process_normalization_data_{fp_name}/`

## How to Run

Run training:
```bash
python tabpfnv2.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
```

Key parameters:
- `--data_name`: Dataset name
- `--task_name`: Task name
- `--task_type`: 'classification' or 'regression'
- `--fp_name`: Fingerprint name
- `--seed`: Random seed
