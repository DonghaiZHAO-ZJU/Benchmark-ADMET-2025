# Evaluating the Use of GNNs and Transfer Learning for Oral Bioavailability Prediction


> ![TOC](https://user-images.githubusercontent.com/66246089/229825030-bf6dbc3f-ecdc-4adb-8c38-2d6427fc5a04.png)

This repository consists of 2 folders and 4 python scripts:

1. data folder
    1. graph_data: csv files of solubility dataset and oral bioavailability dataset
    2. oral_avail_fingerprints: csv files of oral bioavailability dataset fingerprints
    3. oral_mol_desc: csv files of oral bioavailability dataset molecular descriptors 

2. notebooks folder 
    1. Random forest model
    2. GNN from scratch models 
    3. Transfer learning models 
    
3. config.py
    1. hyperparameters and constant variables used in this project

4. engine.py, model.py, utils.py
    1. Contains GNN models and helper functions used in this project 
    
To replicate, please install the dependencies below and follow the instructions in the notebooks.
Please download the saved models from [here](https://drive.google.com/drive/folders/19O4Xo_F-6MKK5H6JE0ykrQ4ZNXoYdOCJ?usp=share_link).

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```


