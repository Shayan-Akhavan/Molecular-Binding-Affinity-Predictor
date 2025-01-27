# Molecular Binding Affinity Predictor 🧬

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2021.03-green.svg)](https://www.rdkit.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A deep learning model that predicts molecular binding affinities using Graph Neural Networks (GNNs) with a range-aware architecture. This project specifically focuses on predicting IC50 values for protein-ligand interactions using data from the ChEMBL database.  

![Image](https://github.com/user-attachments/assets/91e18384-4897-41c5-b639-ac007c07f2b9)



## 📋 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#-evaluation)


## ✨ Features

- 🔬 Graph-based molecular representation using RDKit
- 🧠 Range-aware Graph Neural Network with Graph Attention layers
- 🔄 Automated data fetching from ChEMBL API
- 📊 Stratified sampling across activity ranges
- 📈 Comprehensive evaluation metrics and visualization
- 🎯 Structure-based prediction analysis

## 🚀 Installation

```bash
# Clone the repository
gh repo clone Shayan-Akhavan/molecular-binding-affinity-predictor
cd molecular-binding-affinity-predictor

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
rdkit>=2021.03
torch>=1.9.0
torch-geometric>=2.0.0
pandas>=1.3.0
numpy>=1.19.0
requests>=2.26.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 💻 Usage

### Quick Start

```python
from predictor import fetch_chembl_data, RangeAwareGNN

# Fetch data for EGFR protein target
target_id = "CHEMBL203"
df = fetch_chembl_data(target_id)
df = preprocess_data(df)

# Initialize and train the model
model = RangeAwareGNN(node_features=6, hidden_channels=64)
# Continue with training as shown in main.py
```

### Using Custom Data

```python
from predictor import MolecularGraphDataset

# Prepare your data with 'smiles' and 'activity' columns
dataset = MolecularGraphDataset(df['smiles'].values, df['pActivity'].values)
```
  
## Model Architecture  
  
### Components
- **Input Layer**: Molecular graph representation
  - Nodes: Atoms (atomic number, degree, charge, etc.)
  - Edges: Chemical bonds
- **Hidden Layers**: 3x Graph Attention (GAT) layers with batch normalization
- **Output Layer**: Binding affinity prediction
- **Loss Function**: Custom range-aware loss

<details>
<summary>Detailed Architecture</summary>

```plaintext
Input → GAT Layer 1 → BatchNorm → ReLU → Dropout →
       GAT Layer 2 → BatchNorm → ReLU → Dropout →
       GAT Layer 3 → BatchNorm → ReLU →
       Global Mean Pool → Linear → Output
```
</details>

## 📊 Evaluation

### Metrics
- R² Score
- Mean Absolute Error (MAE)
- Range Coverage Ratio
- Prediction Error Distribution

### Visualization Tools
```python
# Generate performance visualizations
visualize_results(predictions, actuals)

# Analyze prediction outliers
analyze_predictions(smiles_list, predictions, actuals)
```

### Areas for Improvement
- Additional molecular features
- Alternative GNN architectures
- Extended chemical analysis
- Support for more binding metrics
- Ensemble modeling approaches


