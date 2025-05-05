import os
import sys
import torch
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl

# Append root directory to the path to ensure correct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.retain import RETAIN

# Set publication style for plots
def set_publication_style(fontsize=20, label_fontweight='bold', figsize=(12, 8)):
    sns.set_theme(style='whitegrid', palette='viridis')
    mpl.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.labelweight': label_fontweight,
        'xtick.labelsize': fontsize * 0.9,
        'ytick.labelsize': fontsize * 0.9,
        'legend.fontsize': fontsize * 0.9,
        'figure.titlesize': fontsize,
        'figure.titleweight': label_fontweight,
        'figure.figsize': figsize
    })


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load optimized hyperparameters from YAML
def load_best_hyperparameters(hyperparam_path):
    with open(hyperparam_path, 'r') as file:
        return yaml.safe_load(file)

# Load and configure trained RETAIN model with best hyperparameters
def load_trained_model(config, model_path, hyperparam_path):
    # Load best hyperparameters
    best_hyperparams = load_best_hyperparameters(hyperparam_path)

    # Update model config with optimized hyperparameters
    config['model'].update(best_hyperparams)

    # Initialize model
    model = RETAIN(
        input_dim=config['model']['input_dim'],
        emb_dim=config['model']['emb_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=config['model']['device']))
    
    # Move model to specified device and set evaluation mode
    model.to(config['model']['device'])
    model.eval()

    return model

