# **A Comprehensive Benchmark of ADMET Predictors in the Era of Foundation Models**

<figure style="text-align: center; margin: 20px 0;">
    <img src="./figure/Protocol.png" 
         alt="Protocol for the systematic evaluation of ADMET models" 
         style="max-width: 80%; border: 1px solid #eee; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <figcaption style="font-weight: bold; text-align: center; margin-top: 15px; font-size: 1.1em;">
        <strong>Protocol for the systematic evaluation of ADMET models</strong>
    </figcaption>
</figure>

## 🧪 **Datasets Assembling and Preprocessing**

In order to evaluate the potential of various drug efficacy prediction models in practical applications, in this project, we integrated multiple publicly available datasets covering key ADMET (absorption, distribution, metabolism, excretion, and toxicity) prediction endpoints.

Our data processing pipeline involves two key stages: **(1) Data Curation** and **(2) Strategic Data Splitting**.

---

### Datasets Used

We selected a diverse set of datasets to ensure our benchmark is comprehensive. All datasets are provided in the [`data/`](./data/) directory and have been meticulously cleaned, standardized, and deduplicated.

The primary datasets used in this study are summarized below:

| Dataset Name              | Class   | Data Size         | Source / Reference                               |
| ------------------------------ | ---------- | ----------- | ------------------------------------------------ |
| **BBBP** | absorption     | 3873  | `https://doi.org/10.1093/nar/gkab255` `https://doi.org/10.1039/c7sc02664a` `https://doi.org/10.1093/bioinformatics/btaa918`|
| **hERG** | toxicity   | 9673  | `https://doi.org/10.1038/s41467-023-38192-3`     |
| **Mutagenicity** | toxicity                    | 8779    | `https://doi.org/10.1093/nar/gkab255`  `https://doi.org/10.1021/acs.jmedchem.1c00421`                           |
| **Oral Bioavailability** | absorption                    | 1444    | `https://doi.org/10.1021/acs.jcim.3c00554`                             |
| **HLM Metabolic Stability** | metabolism                    | 5860    | `https://doi.org/10.1093/bioinformatics/btad503`                             |
| **Caco2 Permeability** | absorption                    | 842    | `https://doi.org/10.48550/arXiv.2102.09548`                             |
| **HalfLife** | excretion                    | 1314    | `https://doi.org/10.48550/arXiv.2102.09548`                             |
| **VDss** | distribution                    | 1092    | `https://doi.org/10.48550/arXiv.2102.09548`                             |
| **Tox21 NR ER** | toxicity                    | 5964    | `https://doi.org/10.1039/c7sc02664a`                             |
| **Cycpept PAMPA** | absorption                    | 6637    | `https://pubs.acs.org/doi/10.1021/acs.jcim.2c01573`                             |


Please refer to [`data/dataprocessor.py`](data/dataprocessor.py) for the data cleaning and merging process of these datasets.

### Data Splitting Strategies

A critical aspect of evaluating model generalization is the data splitting strategy. To rigorously test the models and simulate real-world scenarios where models must predict on novel chemical matter, we employed several splitting methods beyond a simple random split.

Our splitting strategies include:

* **Random Split**: The standard, baseline approach where data is partitioned randomly. This tests a model's general interpolation ability.

* **Scaffold Split**: This method separates molecules based on their core chemical structure (scaffold). All molecules sharing the same scaffold are placed in the same set (either train or test). This is crucial for testing a model's ability to **generalize to new chemical scaffolds**, which is a more realistic and challenging task.

* **Perimeter Split**: To further stress-test the models, we used the advanced splitter proposed by **Tossou et al. (2024)** in their work on **Real-World Molecular Out-Of-Distribution: Specification and Investigation**. The method creates scene where the test set is intentionally dissimilar from the training set, thereby testing the model's **extrapolation capabilities**.

The implementation for these splitting methods can be found in the [`data/datasplitting.py`](data/datasplitting.py) file. This multi-faceted approach ensures a thorough and robust comparison of the different ADMET predictors.
## 🎯 **ADMET Predictors Evaluation**
We systematically evaluate highly representative drug-likeness prediction models, categorized into two primary technical paradigms:
1. **End-to-End Deep Learning Models**: Directly process molecular graphs/sequences to automatically learn feature representations, suitable for complex nonlinear relationship modeling (GNNs and Transformers)
2. **Feature-Based Classical & Tabular Models**: Rely on expert-engineered molecular descriptors (e.g., ECFP fingerprints, physicochemical properties) and structured tabular data

The complete implementation code and supplementary documentation (including versioning details and custom modifications) are available at:[`selected_models/`](selected_models/).

---

### **Deep Learning Models**
| Model Name              | Class   | Pretrain         | Source / Reference                               |
| ------------------------------ | ---------- | ----------- | ------------------------------------------------ |
| **RGCN** | GNN     | No  | `https://doi.org/10.1038/s41467-023-38192-3`|
| **AttentiveFP** | GNN   | No  | `https://doi.org/10.1021/acs.jmedchem.9b00959`     |
| **D-MPNN** | GNN  | No    | `https://doi.org/10.1021/acs.jcim.3c01250`                           |
| **CMMS-GCL** | GNN   | No    | `https://doi.org/10.1093/bioinformatics/btad503`                             |
| **Vertical-GNN** | GNN    | No    | `https://doi.org/10.1021/acs.jcim.3c00554`                             |
| **GNNAK** | GNN                   | No    | `https://doi.org/10.48550/arXiv.2110.03753`                             |
| **K-BERT** | Transformer                    | Yes    | `https://doi.org/10.1093/bib/bbac131`                             |
| **KPGT** | Graph Transformer                    | Yes    | `https://doi.org/10.1038/s41467-023-43214-1`                             |
| **MOLMCL** | Graph Transformer                  | Yes    | `https://doi.org/10.1038/s41467-024-55082-4`                             |
| **GEM** | 3D GNN                    | Yes    | `https://doi.org/10.1038/s42256-021-00438-4`                             |
| **Uni-Mol** | Graph Transformer                   | Yes    | `https://doi.org/10.26434/chemrxiv-2022-jjm0j-v4`                             |


🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹

### **Classical Algorithms**
| Model Name | Key Feature |
| --- | --- |
| **SVM** | Support Vector Machine, effective in high-dimensional spaces. |
| **KNN** | K-Nearest Neighbors, a non-parametric instance-based learning algorithm. |
| **Random Forest** | Ensemble method using multiple decision trees to improve accuracy. |
| **XGBoost** | Optimized gradient boosting library known for its performance and speed. |
| **LightGBM** | A gradient boosting framework that uses tree-based learning algorithms. |
| **CatBoost** | Gradient boosting on decision trees with good handling of categorical data. |
| **Neural Networks**| Standard feed-forward neural networks (Multi-layer Perceptrons). |


🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹

### **Tabular Models**
| Model Name | Key Feature | Source / Reference |
| --- | --- | --- |
| **AutoGluon** | ​​Automates training, tuning, and ensembling of ML models for tabular data with minimal user effort​. |`https://doi.org/10.48550/arXiv.2003.06505`|
| **TabPFNv2** | Transformer model for tabular data that performs fast inference and adaptation with limited training examples​. |`https://doi.org/10.1038/s41586-024-08328-6`|


💡 **Impact**: TabPFNv2 performs very well in the face of small sample and OOD scenarios.


## 📏 **Roughness Index Application**
---
