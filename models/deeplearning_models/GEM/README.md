# GEM

GEM (GraphEvoM) is a 3D Graph Neural Network with spatial structure-based architecture (GeoGNN) for molecular property prediction.

## Data Representation

GEM uses 3D molecular structures:

- **Input format**: SMILES strings
- **3D features**: Bond-angle graph and atom-bond graph
- **Pretrained model**: Requires pretrained GEM model weights
- **Data files**: CSV files in `./data/raw_data/`

## How to Run

1. **Install dependencies**:
   ```bash
   pip install rdkit torch
   ```

2. **Download pretrained models**: (required)
   - Place pretrained weights in `./pretrain_models-chemrl_gem/`

3. **Run training**:
   ```bash
   bash run.sh
   ```

   Or run directly:
   ```python
   python finetune_class2.py --task_name BBBP \
       --data_name BBBP_random_2024 \
       --data_path ./data/raw_data \
       --processed_data_path ./data/processed_data \
       --compound_encoder_config model_configs/geognn_l8.json \
       --model_config model_configs/down_mlp2.json \
       --init_model ./pretrain_models-chemrl_gem/class.pdparams
   ```

   Key parameters:
   - `--task_name`: Task name
   - `--data_name`: Dataset name with split info
   - `--init_model`: Pretrained model weights (class.pdparams or regr.pdparams)

## Model Architecture

- **Encoder**: GeoGNN with bond-angle graph and atom-bond graph
- **Layers**: 8 graph convolution layers
- **Pretraining tasks**: Bond length, angle, and atomic distance prediction

## References

- Huang, Y. et al. (2021). Gem: An efficient neural network for large-scale molecular property prediction.
