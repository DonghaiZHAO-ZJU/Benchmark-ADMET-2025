![image](https://github.com/user-attachments/assets/bbcb844d-4b7c-463e-8c1a-2580fdd46db4)

# KA-GNN
This is the code of KA-GNN


## Table of Contents
1. [Environment Requirements](#environment-requirements)
2. [Installation Steps](#installation-steps)
3. [Data Download and Configuration](#data-download-and-configuration)
4. [Running the Project](#running-the-project)
5. [Experimental Results](#experimental-results)
## Environment Requirements

This project requires:
- **Python Version**: 3.11
  - Use Python 3.11, as this is the version used for development and testing of the code.
- **CUDA Version**: 11.7
  - To fully utilize GPU acceleration, ensure that your environment supports CUDA 11.7.

## Installation Steps

### 1. Create a Virtual Environment
Recommended to use conda:
```bash
conda create -n myenv python=3.11
conda activate myenv
```

### 2. Install Dependencies
Install the necessary Python libraries from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Verify CUDA Installation
Check that CUDA 11.7 is correctly installed on your system:
```bash
nvcc --version
```

## Data Download and Configuration

### Download Data
Download the datasets from [MoleculeNet](https://moleculenet.org/datasets-1). Place the datasets into the `data` directory in your project folder data.

### Configure Dataset Usage
To use different datasets, modify the `c_path.yaml` file in the `config` directory:
```yaml
select_dataset: "bace"  # Replace "bace" with "hiv" or "muv" as needed
```

## Running the Project

Execute the project with the configured dataset by running:
```bash
python main.py
```

## KA-GNNs(KA-GCN and KA-GAT)
![KA-GNNs](KA_GNN.jpg)  

## Experimental Results

### Comparison with Non-Pretrained Models
The following table presents the comparison of KA-GNN with various GNN architectures. The best performance values are highlighted in **bold**, and standard deviation values are indicated in subscripts.
| Model      | BACE           | BBBP           | ClinTox        | SIDER          | Tox21          | HIV            | MUV            |
|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| No. mol    | 1513           | 2039           | 1478           | 1427           | 7831           | 41127          | 93808          |
| No. avg atoms | 65           | 46             | 50.58          | 65             | 36             | 46             | 43             |
| No. tasks  | 1              | 1              | 2              | 27             | 12             | 1              | 17             |
| D-MPNN     | 0.809_(0.006)  | 0.710_(0.003)  | 0.906_(0.007)  | 0.570_(0.007)  | 0.759_(0.007)  | 0.771_(0.005)  | 0.786_(0.014)  |
| AttentiveFP| 0.784_(0.022)  | 0.663_(0.018)  | 0.847_(0.003)  | 0.606_(0.032)  | 0.781_(0.005)  | 0.757_(0.014)  | 0.786_(0.015)  |
| N-GramRF   | 0.779_(0.015)  | 0.697_(0.006)  | 0.775_(0.040)  | 0.668_(0.007)  | 0.743_(0.009)  | 0.772_(0.004)  | 0.769_(0.002)  |
| N-GramXGB  | 0.791_(0.013)  | 0.691_(0.008)  | 0.875_(0.027)  | 0.655_(0.007)  | 0.758_(0.009)  | 0.787_(0.004)  | 0.748_(0.002)  |
| PretrainGNN| 0.845_(0.007)  | 0.687_(0.013)  | 0.726_(0.015)  | 0.627_(0.008)  | 0.781_(0.006)  | 0.799_(0.007)  | 0.813_(0.021)  |
| GROVE_base | 0.821_(0.007)  | 0.700_(0.001)  | 0.812_(0.030)  | 0.648_(0.006)  | 0.743_(0.001)  | 0.625_(0.009)  | 0.673_(0.018)  |
| GROVE_large| 0.810_(0.014)  | 0.695_(0.001)  | 0.762_(0.037)  | 0.654_(0.001)  | 0.735_(0.001)  | 0.682_(0.011)  | 0.673_(0.018)  |
| GraphMVP   | 0.812_(0.009)  | 0.724_(0.016)  | 0.791_(0.028)  | 0.639_(0.012)  | 0.759_(0.005)  | 0.770_(0.012)  | 0.777_(0.006)  |
| MolCLR     | 0.824_(0.009)  | 0.722_(0.021)  | 0.912_(0.035)  | 0.589_(0.014)  | 0.750_(0.002)  | 0.781_(0.005)  | 0.796_(0.019)  |
| GEM        | 0.856_(0.011)  | 0.724_(0.004)  | 0.901_(0.013)  | 0.672_(0.004)  | 0.781_(0.001)  | 0.806_(0.009)  | 0.817_(0.005)  |
| Mol-GDL    | 0.863_(0.019)  | 0.728_(0.019)  | 0.966_(0.002)  | 0.831_(0.002)  | 0.794_(0.005)  | 0.808_(0.007)  | 0.675_(0.014)  |
| Uni-mol    | 0.857_(0.002)  | 0.729_(0.006)  | 0.919_(0.018)  | 0.659_(0.013)  | 0.796_(0.005)  | 0.808_(0.003)  | 0.821_(0.013)  |
| SMPT       | 0.873_(0.015)  | 0.734_(0.003)  | 0.927_(0.002)  | 0.676_(0.050)  | 0.797_(0.001)  | 0.812_(0.001)  | 0.822_(0.008)  |
| **KA-GCN** | **0.890_(0.014)** | **0.787_(0.014)** | **0.989_(0.003)** | **0.842_(0.001)** | **0.799_(0.005)** | **0.821_(0.005)** | **0.834_(0.009)**|
| **KA-GAT** | **0.884_(0.004)** | **0.785_(0.021)** | **0.991_(0.005)** | **0.847_(0.002)** | **0.800_(0.006)** | **0.823_(0.002)** | **0.834_(0.010)**|

## Citing

If you find our codes useful in your research, please consider citing:

```bibtex
@article{li2025kolmogorov,
  title={Kolmogorov--Arnold graph neural networks for molecular property prediction},
  author={Li, Longlong and Zhang, Yipeng and Wang, Guanghui and Xia, Kelin},
  journal={Nature Machine Intelligence},
  pages={1--9},
  year={2025},
  publisher={Nature Publishing Group}
}
