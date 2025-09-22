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
│   │   ├── transformer.py       # Transformer baseline model
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
│   ├── transformer_config.yaml  # Transformer model configuration
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

**Easy dataset selection with `--dataset` flag:**

Train all models on CKD dataset (results saved to `./results/ckd/`):
```bash
python scripts/train.py --model bilstm --dataset ckd --hyperparameter-search
python scripts/train.py --model retain --dataset ckd --hyperparameter-search
python scripts/train.py --model transformer --dataset ckd --hyperparameter-search
python scripts/train.py --model adacare --dataset ckd --hyperparameter-search
python scripts/train.py --model stagenet --dataset ckd --hyperparameter-search
python scripts/train.py --model coi --dataset ckd --hyperparameter-search
```

Train all models on MIMIC-IV dataset (results saved to `./results/mimic/`):
```bash
python scripts/train.py --model bilstm --dataset mimic --hyperparameter-search
python scripts/train.py --model retain --dataset mimic --hyperparameter-search
python scripts/train.py --model transformer --dataset mimic --hyperparameter-search
python scripts/train.py --model adacare --dataset mimic --hyperparameter-search
python scripts/train.py --model stagenet --dataset mimic --hyperparameter-search
python scripts/train.py --model coi --dataset mimic --hyperparameter-search
```

**Alternative: Manual config specification:**
```bash
# You can still specify config files manually if needed
python scripts/train.py --model transformer --config config/mimiciv_transformer_config.yaml --hyperparameter-search
```

**Automated Complete Pipeline:**

Run all models on both datasets automatically:

**Linux/Mac:**
```bash
# Make script executable and run
chmod +x run_all_experiments.sh
./run_all_experiments.sh

# Or with custom parameters
./run_all_experiments.sh --combinations 10 --conda-env coi
```

**Windows:**
```cmd
# Run with default settings (2 combinations per model)
run_all_experiments.bat

# Or with custom parameters (combinations, conda-env)
run_all_experiments.bat 10 coi
```

This pipeline will:
1. Train all 6 models (BiLSTM, RETAIN, Transformer, AdaCare, StageNet, CoI) on both datasets
2. Automatically generate comprehensive summary tables
3. Provide progress tracking and error handling
4. Show total time and success statistics

**Manual Results Analysis:**

If you want to analyze results separately:
```bash
# Analyze all results and create summary tables
python scripts/analyze_results.py
```

This will generate:
- `./results/summary_ckd_results.csv` - CKD dataset results table
- `./results/summary_mimic_results.csv` - MIMIC-IV dataset results table
- `./results/combined_results_summary.xlsx` - Combined Excel file with both datasets
- `./results/results_analysis_summary.json` - Analysis summary with best performing models

**Model Analysis:**

Generate comprehensive analysis and visualizations:

```bash
# Analyze trained models
python scripts/analyze.py --model coi --output results/coi_analysis

# Generate comparison figures
python scripts/analyze.py --model retain --output results/comparison
python scripts/analyze.py --model coi --output results/comparison
```

**Hyperparameter Tuning:**

Control the number of hyperparameter combinations to test:

```bash
# Quick test (2-5 minutes)
python scripts/train.py --model transformer --dataset ckd --hyperparameter-search --n-combinations 2

# Development testing (10-30 minutes)
python scripts/train.py --model transformer --dataset ckd --hyperparameter-search --n-combinations 5

# Thorough tuning (1-5 hours)
python scripts/train.py --model transformer --dataset ckd --hyperparameter-search --n-combinations 20

# Default: 300 combinations (comprehensive search)
python scripts/train.py --model transformer --dataset ckd --hyperparameter-search
```

Available hyperparameter combinations per model:
- **Transformer**: 1,458 total combinations (emb_dim×hidden_dim×num_heads×num_layers×lr×batch_size×dropout)
- **CoI**: 972 total combinations
- **RETAIN**: 324 total combinations
- **BiLSTM**: 108 total combinations

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

### Transformer
Modern attention-based baseline using standard transformer architecture:
- **Multi-Head Self-Attention**: Captures complex feature interactions across all time steps
- **Positional Encoding**: Incorporates temporal position information
- **Parallel Processing**: Efficient computation compared to sequential RNN models
- **Two Variants**: Standard pooling-based and CLS token-based approaches

### AdaCare
Recent RETAIN-based model with adaptive feature calibration (2020):
- **Feature Calibration**: Adaptive recalibration of feature importance based on global and local contexts
- **Enhanced Interpretability**: Improved attention mechanisms for better clinical explanation
- **Knowledge Distillation**: Advanced representation learning for better performance
- **Multi-level Attention**: Combines temporal and feature-level attention with calibration

### StageNet
Stage-aware neural network for health risk prediction (2020):
- **Disease Stage Modeling**: Explicitly models different stages of disease progression
- **Stage-aware Convolution**: Uses different kernels to capture stage-specific patterns
- **Temporal Stage Transitions**: Models how patients transition between disease stages
- **Adaptive Stage Attention**: Combines temporal attention with stage-aware weighting

## Experimental Results

### Performance Comparison on CKD Dataset

| Model | AUROC | F1-Score | Accuracy | Precision |
|-------|-------|----------|----------|-----------|
| BiLSTM | 0.930 | 0.650 | 0.910 | 0.750 |
| RETAIN | 0.930 | 0.660 | 0.920 | 0.760 |
| Transformer | TBD | TBD | TBD | TBD |
| AdaCare | TBD | TBD | TBD | TBD |
| StageNet | TBD | TBD | TBD | TBD |
| **CoI** | **0.950** | **0.690** | **0.940** | **0.790** |

*Note: Results for new baseline models will be updated after training completion.*

### Key Findings

- **Superior AUROC Performance**: CoI achieves 0.950 AUROC, outperforming both RETAIN and BiLSTM (both 0.930) by 2.0 percentage points
- **Best F1-Score**: CoI demonstrates the highest F1-score (0.690), with improvements of 3.0 points over RETAIN (0.660) and 4.0 points over BiLSTM (0.650)
- **Highest Accuracy**: CoI achieves 94.0% accuracy, surpassing RETAIN (92.0%) by 2.0 points and BiLSTM (91.0%) by 3.0 points
- **Improved Precision**: CoI shows the best precision (0.790), outperforming RETAIN (0.760) by 3.0 points and BiLSTM (0.750) by 4.0 points
- **Clinical Interpretability**: The model successfully identifies clinically meaningful influence chains, such as eGFR → Hemoglobin → Healthcare utilization patterns



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
