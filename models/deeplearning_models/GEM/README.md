# GEM

GEM is a 3D Graph Neural Network with spatial structure-based architecture (GeoGNN) for molecular property prediction.

## Data Representation

- **Input**: SMILES strings
- **3D features**: Bond-angle graph and atom-bond graph
- **Data format**: CSV files

## How to Run

Run training:
```bash
bash run.sh
```

Or run with Python:
```bash
python finetune_class2.py --task_name BBBP \
    --data_name BBBP_random_2024 \
    --data_path ./data/raw_data \
    --processed_data_path ./data/processed_data \
    --init_model ./pretrain_models-chemrl_gem/class.pdparams
```

Key parameters:
- `--task_name`: Task name
- `--data_name`: Dataset name with split info
- `--init_model`: Pretrained model weights (class.pdparams or regr.pdparams)
