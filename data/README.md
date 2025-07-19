## ⚙️ Setup and Installation

This project involves running two main Python scripts: `dataprocessor.py` and `datasplitting.py`.

### Dependencies

The `datasplitting.py` script requires the **mood** library. For installation instructions and further details, please refer to the official repository:

[https://github.com/valence-labs/mood-experiments](https://github.com/valence-labs/mood-experiments?tab=readme-ov-file)

<hr>

### Running the Scripts

Once the dependencies are installed, run the scripts in the following order.

**1. Clean the Dataset**

The `dataprocessor.py` script is used to clean the raw dataset and remove any duplicate entries.

```bash
python dataprocessor.py
```

**2. Split the Dataset**

After cleaning, the `datasplitting.py` script partitions the dataset using various splitting methods(**Random Split, Scaffold Split and Perimeter Split**).

```bash
python datasplitting.py
```