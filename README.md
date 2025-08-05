# Chain-of-Influence: Tracing Interdependencies Across Time and Features in Clinical Predictive Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A novel interpretable deep learning framework for clinical time series prediction that explicitly models temporal-feature interdependencies.

## Overview

Chain-of-Influence (CoI) addresses the critical gap in clinical predictive modeling by capturing how features influence each other across time. Unlike traditional approaches that treat features independently, CoI provides unprecedented transparency into the complex chains of influence that drive clinical outcomes.

<div align="center">
  <img src="assets/influence_network.png" alt="Chain-of-Influence Network Visualization" width="800"/>
  <p><em>Chain-of-Influence network showing temporal-feature interdependencies in CKD progression. Node size reflects feature importance, arrow thickness indicates influence strength, and colors distinguish influence types.</em></p>
</div>

### Key Features

- **Explicit Influence Modeling**: Traces how feature A at time t affects feature B at time t+k, and finally to predictions, showing how effects propagate as a chain of influence
- **Multi-level Attention**: Combines temporal attention with cross-feature interactions
- **Comprehensive Interpretability**: Provides feature importance and influence chain analysis
- **Interactive Demonstration**: Explore the model with our comprehensive Jupyter notebook
- **Clinical Validation**: Evaluated on chronic CKD progression prediction and acute ICU mortality prediction tasks

## Repository Structure

```
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── bilstm.py            # Bidirectional LSTM baseline
│   │   ├── retain.py            # RETAIN attention model
│   │   └── coi.py               # Chain-of-Influence model
│   ├── data/                    # Data processing utilities
│   │   ├── preprocess.py        # Data preprocessing
│   │   └── tsmote.py            # Temporal SMOTE implementation
│   └── utils/                   # Utility functions
│       └── report_best_model.py # Model reporting utilities
├── scripts/                     # Training and analysis scripts
│   ├── train.py                 # Model training script
│   ├── analyze.py               # Model analysis and visualization
│   └── preprocess_mimiciv.py    # MIMIC-IV preprocessing
├── config/                      # Configuration files
│   ├── coi_config.yaml          # CoI model configuration
│   ├── retain_config.yaml       # RETAIN model configuration
│   ├── bilstm_config.yaml       # BiLSTM model configuration
│   ├── mimiciv_*.yaml          # MIMIC-IV specific configs
│   └── feature_names.yaml      # Feature specifications
├── assets/                      # Visual assets and interactive demonstrations
│   ├── Chain_of_Influence_demo.ipynb # Interactive demonstration notebook
│   ├── sample_influence_matrix.csv # Sample test data for demos
│   ├── README.md               # Demo documentation
│   └── influence_network.png    # Chain-of-Influence network visualization
├── data/                        # Data directory (see data/README.md)
│   └── README.md               # Data acquisition instructions
└── results/                     # Experimental results
    ├── mimic/                   # MIMIC-IV results
    └── ckd/                     # CKD results
```

## Quick Start

### Interactive Demo

Experience Chain-of-Influence with our comprehensive interactive demonstration notebook: [`assets/Chain_of_Influence_demo.ipynb`](assets/Chain_of_Influence_demo.ipynb)

- **File Upload Interface**: Drag-and-drop support for influence matrices (CSV/Excel/JSON)
- **Interactive Network Visualization**: Physics-based layouts with real-time parameter adjustment
- **AI-Powered Analysis**: GPT-4 integration for pattern recognition and comprehensive analysis of modeling results
- **Professional Dashboard**: Six comprehensive tabs covering model performance, key timestamps, key features, cross-temporal analysis, LLM-generated summaries, and token usage tracking

### Development Setup

For researchers and developers who want to train models or reproduce results:

<details>
<summary><strong>Click to expand local development setup</strong></summary>

**Installation:**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chain-of-influence
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate coi
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

**Data Setup:**

Follow the instructions in [`data/README.md`](data/README.md) to obtain and organize the datasets:
- **CKD Dataset**: Private longitudinal clinical data (access restricted for privacy)
- **MIMIC-IV v3.1**: Public critical care database (requires PhysioNet credentialed access)

**Training Models:**

Train all three models on the CKD dataset:

```bash
# Train BiLSTM baseline
python scripts/train.py --model bilstm --hyperparameter-search

# Train RETAIN model
python scripts/train.py --model retain --hyperparameter-search

# Train Chain-of-Influence model
python scripts/train.py --model coi --hyperparameter-search
```

For MIMIC-IV dataset:
```bash
python scripts/train.py --model coi --config config/mimiciv_coi_config.yaml --hyperparameter-search
```

**Model Analysis:**

Generate comprehensive analysis and visualizations:

```bash
# Analyze trained models
python scripts/analyze.py --model coi --output results/coi_analysis

# Generate comparison figures
python scripts/analyze.py --model retain --output results/comparison
python scripts/analyze.py --model coi --output results/comparison
```

</details>

## Models

### Chain-of-Influence (CoI)
Our novel architecture that explicitly models temporal-feature interdependencies:
- **Temporal Attention**: Identifies critical time periods in disease progression
- **Cross-Feature Attention**: Captures feature interactions across different time steps
- **Dynamic Tanh (DyT)**: Replaces traditional normalization layers for improved stability
- **Interpretability Framework**: Provides comprehensive influence chain visualization and analysis

### RETAIN
Attention-based baseline model with dual-level attention mechanism:
- **Visit-level Attention**: Assigns importance weights to different time steps
- **Variable-level Attention**: Determines feature importance within each time step
- **Reverse Time Processing**: Processes sequences in reverse chronological order to mimic clinical reasoning

### Bidirectional LSTM
Strong recurrent neural network baseline:
- **Bidirectional Processing**: Captures both forward and backward temporal dependencies
- **Simple Architecture**: Provides robust performance comparison baseline without attention mechanisms

## Experimental Results

### Performance Comparison on CKD Dataset

| Model | AUROC | F1-Score | Accuracy | Precision |
|-------|-------|----------|----------|-----------|
| BiLSTM | 0.930 | 0.650 | 0.910 | 0.750 |
| RETAIN | 0.930 | 0.660 | 0.920 | 0.760 |
| **CoI** | **0.950** | **0.690** | **0.940** | **0.790** |

### Key Findings

- **Consistent Performance Gains**: CoI demonstrates superior performance across all evaluation metrics compared to both baseline models
- **Clinical Interpretability**: The model successfully identifies clinically meaningful influence chains, such as eGFR → Hemoglobin → Healthcare utilization patterns
- **Cross-Domain Validation**: Strong performance demonstrated on both chronic disease progression (CKD) and acute care prediction (MIMIC-IV mortality) tasks
- **High Precision**: Achieves perfect precision (1.000) on MIMIC-IV in-hospital mortality prediction



## Contributing

We welcome contributions from the research community. Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description of changes
4. Ensure all tests pass and follow coding standards

For bug reports or feature requests, please open an issue with detailed information.


## Acknowledgments

- **MIMIC-IV Database**: MIT Laboratory for Computational Physiology for providing critical care data
- **Clinical Collaborators**: Domain experts who provided validation and clinical insight
- **Research Community**: Reviewers and colleagues who provided constructive feedback

## Contact

- Available upon acceptance
