# CLAUDE.md

## Run with ka-gnn
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ka-gnn
python main.py --select_dataset bace --loss_sclect bce --split_method scaffold --seed 2024
```

## Datasets
- Classification: bce loss (BBBP, BACE, ClinTox, SIDER, Tox21, HIV, MUV)
- Regression: l2 loss

## Split Methods
- random, scaffold, Perimeter
