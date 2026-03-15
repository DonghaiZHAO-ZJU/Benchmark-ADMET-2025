# TabPFNv2

TabPFNv2 is a pretrained Transformer-based foundation model for tabular data, designed for fast inference and adaptation with limited training examples.

## Data Representation

TabPFNv2 uses molecular fingerprints:

- **Input format**: CSV files with SMILES and labels
- **Features**: Molecular fingerprints (Morgan, MACCS, etc.)
- **Data location**: `./data/process_normalization_data_{fp_name}/`

## How to Run

1. **Install dependencies**:
   ```bash
   pip install tabpfn rdkit pandas numpy
   ```

2. **Run training**:
   ```bash
   python tabpfnv2.py --data_name BBBP --task_name BBBP --task_type classification --fp_name MorganFP --seed 2024
   ```

   Key parameters:
   - `--data_name`: Dataset name
   - `--task_name`: Task name
   - `--task_type`: 'classification' or 'regression'
   - `--fp_name`: Fingerprint name
   - `--seed`: Random seed

3. **Other variants**:
   - `tabpfnv2-hpo.py`: With hyperparameter optimization
   - `tabpfnv2-time.py`: With time-based evaluation

## Model Architecture

- **Architecture**: Transformer-based foundation model
- **Pretraining**: Trained on synthetic datasets generated from structural causal models
- **Inference**: Fast single forward pass for entire dataset
- **Adaptation**: Works well with limited training data

## Key Features

- **Foundation model**: Pretrained on diverse tabular data
- **Few-shot learning**: Excellent performance with small datasets
- **OOD robustness**: Good generalization to out-of-distribution data
- **Fast inference**: Processes entire dataset in one forward pass

## References

- Hollmann, N. et al. (2024). TabPFNv2: Powerful General-Purpose Tabular Foundation Model. Nature.
