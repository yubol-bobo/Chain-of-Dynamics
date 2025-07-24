#!/usr/bin/env python3
"""
Unified Training Script for RETAIN and TFCAM Models

This script provides a unified training interface for both RETAIN and TFCAM models,
with support for hyperparameter tuning, TSMOTE data balancing, and comprehensive
training metrics.
"""

import os
import sys
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, precision_score, f1_score
from datetime import datetime
import itertools
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils to path
sys.path.append(os.path.dirname(__file__))
from tsmote import TSMOTE

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.retain import RETAIN
from models.tfcam import TFCAM
from models.enhanced_tfcam import EnhancedTFCAM
from models.hcta import HCTA
from models.ctga import CTGA
from models.mstca import MSTCA

def get_shape(x):
    if hasattr(x, 'shape'):
        return x.shape
    elif hasattr(x, '__len__'):
        return (len(x),)
    else:
        return 'unknown'

def generate_timestamp():
    """Generate timestamp in format YYYYMMDD_HHMMSS"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_best_params(params_path):
    """Load best hyperparameters from file."""
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def update_config_with_timestamp(config):
    """Update config with current timestamp"""
    timestamp = generate_timestamp()
    config['timestamp'] = timestamp
    print(f"✅ Generated timestamp: {timestamp}")
    return config

def get_best_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def prepare_data(config):
    """
    Prepare data for training with TSMOTE balancing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_data, val_data, test_data: TensorDataset objects
    """
    print("Preparing data...")
    data = pd.read_csv(config['data']['processed_path']).fillna(0)
    shape_0 = data.shape[0]
    print(f"✅ Data loaded: {data.shape}")
    
    input_dim = config['model']['input_dim']
    num_period = config['data']['month_count'] // 3
    print(f"✅ Input dim: {input_dim}, Num periods: {num_period}")
    
    # Extract features and labels
    features = data.drop(columns=['TMA_Acct', 'ESRD'])
    labels = data['ESRD']
    print(f"✅ Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    # Clean features: replace NaN and Inf with 0.0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Reshape the data to n_patient x timesteps x input_dim
    features_reshaped = features_normalized.reshape(shape_0, num_period, input_dim)

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features_reshaped, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.float32)

    # Split the dataset into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_tensor, labels_tensor, test_size=0.4, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    # Ensure all splits are numpy arrays before converting to tensors
    def to_tensor(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return torch.as_tensor(x, dtype=torch.float32)

    X_train = to_tensor(X_train)
    X_val = to_tensor(X_val)
    X_test = to_tensor(X_test)
    y_train = to_tensor(y_train)
    y_val = to_tensor(y_val)
    y_test = to_tensor(y_test)

    print(f"✅ Data split - Train: {get_shape(X_train)}, Val: {get_shape(X_val)}, Test: {get_shape(X_test)}")  
    
    # Create Tensor datasets
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    # Convert PyTorch tensors to numpy arrays for TSMOTE
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()

    # Apply TSMOTE for data balancing
    tsmote = TSMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = tsmote.fit_resample(X_train_np, y_train_np)

    # Convert the resampled data back to PyTorch tensors
    X_train_tensor_resampled = to_tensor(X_train_resampled)
    y_train_tensor_resampled = to_tensor(y_train_resampled)
    train_data = TensorDataset(X_train_tensor_resampled, y_train_tensor_resampled)

    return train_data, val_data, test_data

def create_model(config):
    """Create model based on config type"""
    model_type = config['model'].get('type', 'retain')
    
    if model_type.lower() == 'tfcam':
        model = TFCAM(
            input_dim=config['model']['input_dim'],
            emb_dim=config['model']['emb_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            output_dim=config['model']['output_dim'],
            dropout=config['model'].get('dropout', 0.2),
            max_seq_len=config['model'].get('max_seq_len', 50)
        )
        
        # Set DyT alpha initialization if specified
        if 'alpha_init' in config['model']:
            for module in model.modules():
                if hasattr(module, 'alpha') and isinstance(module.alpha, torch.nn.Parameter):
                    module.alpha.data = torch.ones_like(module.alpha) * config['model']['alpha_init']
    elif model_type.lower() == 'enhanced_tfcam':
        model = EnhancedTFCAM(
            input_dim=config['model']['input_dim'],
            emb_dim=config['model']['emb_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            output_dim=config['model']['output_dim'],
            dropout=config['model'].get('dropout', 0.2),
            max_seq_len=config['model'].get('max_seq_len', 50)
        )
        # Set DyT alpha initialization if specified
        if 'alpha_init' in config['model']:
            for module in model.modules():
                if hasattr(module, 'alpha') and isinstance(module.alpha, torch.nn.Parameter):
                    module.alpha.data = torch.ones_like(module.alpha) * config['model']['alpha_init']
    elif model_type.lower() == 'hcta':
        model = HCTA(
            input_dim=config['model']['input_dim'],
            emb_dim=config['model']['emb_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            output_dim=config['model']['output_dim'],
            dropout=config['model'].get('dropout', 0.2),
            max_seq_len=config['model'].get('max_seq_len', 50)
        )
        # Set DyT alpha initialization if specified
        if 'alpha_init' in config['model']:
            for module in model.modules():
                if hasattr(module, 'alpha') and isinstance(module.alpha, torch.nn.Parameter):
                    module.alpha.data = torch.ones_like(module.alpha) * config['model']['alpha_init']
    elif model_type.lower() == 'ctga':
        model = CTGA(
            input_dim=config['model']['input_dim'],
            emb_dim=config['model']['emb_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            output_dim=config['model']['output_dim'],
            dropout=config['model'].get('dropout', 0.2),
            max_seq_len=config['model'].get('max_seq_len', 50)
        )
        # Set DyT alpha initialization if specified
        if 'alpha_init' in config['model']:
            for module in model.modules():
                if hasattr(module, 'alpha') and isinstance(module.alpha, torch.nn.Parameter):
                    module.alpha.data = torch.ones_like(module.alpha) * config['model']['alpha_init']
    elif model_type.lower() == 'mstca':
        model = MSTCA(
            input_dim=config['model']['input_dim'],
            emb_dim=config['model']['emb_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            output_dim=config['model']['output_dim'],
            dropout=config['model'].get('dropout', 0.2),
            max_seq_len=config['model'].get('max_seq_len', 50)
        )
        # Set DyT alpha initialization if specified
        if 'alpha_init' in config['model']:
            for module in model.modules():
                if hasattr(module, 'alpha') and isinstance(module.alpha, torch.nn.Parameter):
                    module.alpha.data = torch.ones_like(module.alpha) * config['model']['alpha_init']
    else:  # Default to RETAIN
        model = RETAIN(
            input_dim=config['model']['input_dim'],
            emb_dim=config['model']['emb_dim'],
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            dropout=config['model'].get('dropout', 0.2)
        )
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, config, log_alpha=False):
    """
    Train model with early stopping and comprehensive metrics.
    Early stopping is now based on validation loss instead of F1 score.
    """
    print("Training model...")
    device = torch.device(config['model']['device'])
    model.to(device)
    best_val_loss, trigger_times = float('inf'), 0
    patience, n_epochs = config['model']['patience'], config['model']['n_epochs']
    save_path = config['paths']['save_path']
    model_type = config['model'].get('type', 'retain')

    metrics = {
        'train_losses': [], 'val_losses': [], 'val_aucs': [],
        'val_recalls': [], 'val_accuracies': [], 'val_precisions': [], 'val_f1s': []
    }

    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        epoch_losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        metrics['train_losses'].append(np.mean(epoch_losses))
        model.eval()
        val_loss_epoch, y_true, y_pred = [], [], []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss_epoch.append(criterion(outputs, labels.unsqueeze(1)).item())
                predicted_probs = torch.sigmoid(outputs).view(-1)
                y_pred.extend(predicted_probs.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
                # Debug: print first batch input, label, and output if NaNs are detected
                if batch_idx == 0 and (torch.isnan(outputs).any() or torch.isnan(inputs).any()):
                    print("[DEBUG] First validation batch input (sample):", inputs.flatten()[:10].cpu().numpy())
                    print("[DEBUG] First validation batch label (sample):", labels.flatten()[:10].cpu().numpy())
                    print("[DEBUG] First validation batch model output (pre-sigmoid, sample):", outputs.flatten()[:10].cpu().numpy())

        binary_preds = (np.array(y_pred) > 0.5).astype(int)
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        if np.isnan(y_true_np).any() or np.isnan(y_pred_np).any():
            print("[WARNING] NaN detected in validation labels or predictions!")
            print(f"y_true sample: {y_true_np[:10]}")
            print(f"y_pred sample: {y_pred_np[:10]}")
            print("Skipping metric calculation for this epoch.")
            metrics['val_losses'].append(np.mean(val_loss_epoch))
            metrics['val_aucs'].append(np.nan)
            metrics['val_recalls'].append(np.nan)
            metrics['val_accuracies'].append(np.nan)
            metrics['val_precisions'].append(np.nan)
            metrics['val_f1s'].append(np.nan)
            continue
        val_f1 = f1_score(y_true_np, binary_preds)
        current_val_loss = np.mean(val_loss_epoch)
        metrics['val_losses'].append(current_val_loss)
        metrics['val_aucs'].append(roc_auc_score(y_true_np, y_pred_np))
        metrics['val_recalls'].append(recall_score(y_true_np, binary_preds))
        metrics['val_accuracies'].append(accuracy_score(y_true_np, binary_preds))
        metrics['val_precisions'].append(precision_score(y_true_np, binary_preds))
        metrics['val_f1s'].append(val_f1)

        # print(f"Epoch [{epoch+1}/{n_epochs}] Train Loss: {metrics['train_losses'][-1]:.4f} "
              #f"Val Loss: {metrics['val_losses'][-1]:.4f} Val F1: {val_f1:.4f}")

        # Optional: Log DyT alpha values for TFCAM
        if log_alpha and model_type.lower() == 'tfcam':
            alpha_values = []
            for module in model.modules():
                if hasattr(module, 'alpha') and isinstance(module.alpha, torch.nn.Parameter):
                    alpha_values.append(module.alpha.item())
            if alpha_values:
                print(f"DyT alpha values: {alpha_values}")

        # Early stopping based on validation loss (lower is better)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            trigger_times = 0
            timestamp = config.get('timestamp', 'unknown')
            model_filename = f"{model_type}_best_model_{timestamp}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparams': config['model']
            }, os.path.join(save_path, model_filename))
            # print(f"✅ Best model updated with Val Loss: {best_val_loss:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("🛑 Early stopping triggered (validation loss not improving).")
                break
                
    return metrics

def hyperparameter_search(config, train_data, val_data, test_data, n_combinations=10):
    model_type = config['model'].get('type', 'retain')
    device = torch.device(config['model']['device'])
    save_path = config['paths']['save_path']
    if model_type.lower() == 'tfcam':
        hyperparameters = {
            'emb_dim': [16, 32, 64],
            'hidden_dim': [32, 64, 128],
            'num_heads': [2, 4, 8],
            'num_layers': [2, 4, 8],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [64, 128],
            'dropout': [0, 0.2, 0.4],
            'alpha_init': [0.6, 0.8, 1.0]
        }
    elif model_type.lower() == 'enhanced_tfcam':
        hyperparameters = {
            'emb_dim': [16, 32, 64],
            'hidden_dim': [32, 64, 128],
            'num_heads': [2, 4, 8],
            'num_layers': [2, 4, 8],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [64, 128],
            'dropout': [0, 0.2, 0.4],
            'alpha_init': [0.6, 0.8, 1.0]
        }
    elif model_type.lower() == 'hcta':
        hyperparameters = {
            'emb_dim': [16, 32, 64],
            'hidden_dim': [32, 64, 128],
            'num_heads': [2, 4, 8],
            'num_layers': [2, 4, 8],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [64, 128],
            'dropout': [0, 0.2, 0.4],
            'alpha_init': [0.6, 0.8, 1.0]
        }
    elif model_type.lower() == 'ctga':
        hyperparameters = {
            'emb_dim': [16, 32, 64],
            'hidden_dim': [32, 64, 128],
            'num_heads': [2, 4, 8],
            'num_layers': [2, 4, 8],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [64, 128],
            'dropout': [0, 0.2, 0.4],
            'alpha_init': [0.6, 0.8, 1.0]
        }
    elif model_type.lower() == 'mstca':
        hyperparameters = {
            'emb_dim': [16, 32, 64],
            'hidden_dim': [32, 64, 128],
            'num_heads': [2, 4, 8],
            'num_layers': [2, 4, 8],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [64, 128],
            'dropout': [0, 0.2, 0.4],
            'alpha_init': [0.6, 0.8, 1.0]
        }
    else:
        hyperparameters = {
            'emb_dim': [16, 32, 64],
            'hidden_dim': [32, 64, 128],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [64, 128],
            'dropout': [0, 0.2, 0.4]
        }
    param_names = list(hyperparameters.keys())
    param_values = list(hyperparameters.values())
    all_combinations = list(itertools.product(*param_values))
    if len(all_combinations) > n_combinations:
        all_combinations = random.sample(all_combinations, n_combinations)
    best_val_loss, best_f1, best_auc, best_hyperparams, best_model_path = float('inf'), -np.inf, -np.inf, None, None
    criterion = nn.BCEWithLogitsLoss()
    print(f"🔍 Starting hyperparameter search for {model_type.upper()}...")
    print(f"📊 Testing {len(all_combinations)} combinations (full training each)...")
    for i, params in enumerate(tqdm(all_combinations, desc="Hyperparameter Search")):
        params_dict = dict(zip(param_names, params))
        config['model'].update(params_dict)
        model = create_model(config)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=params_dict['learning_rate'])
        train_loader = DataLoader(train_data, batch_size=params_dict['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=params_dict['batch_size'])
        metrics = train_model(model, train_loader, val_loader, criterion, optimizer, config)
        val_losses = metrics['val_losses']
        best_epoch = int(np.argmin(val_losses))
        current_val_loss = val_losses[best_epoch]
        current_f1 = metrics['val_f1s'][best_epoch]
        current_auc = metrics['val_aucs'][best_epoch]
        model_filename = f"{model_type}_best_model_hypersearch.pt"
        candidate_model_path = os.path.join(save_path, model_filename)
        # Save model based on validation loss (lower is better)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_f1 = current_f1
            best_auc = current_auc
            best_hyperparams = params_dict
            best_model_path = candidate_model_path
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparams': config['model']
            }, candidate_model_path)
            print(f"🎯 New best model (Val Loss: {best_val_loss:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}) with params: {params_dict}")
    # Save summary file
    import yaml
    summary = {
        "best_val_loss": float(best_val_loss),
        "best_f1": float(best_f1),
        "best_auc": float(best_auc),
        "best_params": best_hyperparams,
        "model_path": best_model_path,
        "timestamp": config.get("timestamp", "")
    }
    summary_path = os.path.join(save_path, f"{model_type}_hypersearch_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f)
    print(f"✅ Hyperparameter search summary saved to: {summary_path}")
    return best_hyperparams, best_f1, best_auc, best_model_path

def save_training_results(metrics, config):
    """Save training results with timestamp"""
    timestamp = config.get('timestamp', 'unknown')
    results_path = config['paths']['results_path']
    
    # Save metrics
    metrics_filename = f"training_metrics_{timestamp}.yaml"
    metrics_path = os.path.join(results_path, metrics_filename)
    
    # Convert numpy arrays to lists for YAML serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        else:
            metrics_serializable[key] = value
    
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics_serializable, f)
    
    print(f"✅ Training metrics saved to: {metrics_path}")

def save_best_params(best_params, config, timestamp):
    """Save best hyperparameters"""
    results_path = config['paths']['results_path']
    best_params_path = os.path.join(results_path, f'best_hyperparameters_{timestamp}.yaml')
    
    with open(best_params_path, 'w') as f:
        yaml.dump(best_params, f)
    
    print(f"✅ Best hyperparameters saved to: {best_params_path}")

def plot_training_metrics(metrics, config):
    """Plot training metrics"""
    timestamp = config.get('timestamp', 'unknown')
    results_path = config['paths']['results_path']
    model_type = config['model'].get('type', 'retain')
    
    # Create plots directory
    plots_dir = os.path.join(results_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{model_type.upper()} Training Metrics', fontsize=16)
    
    # Plot losses
    axes[0, 0].plot(metrics['train_losses'], label='Train Loss')
    axes[0, 0].plot(metrics['val_losses'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # Plot AUC
    axes[0, 1].plot(metrics['val_aucs'], label='Val AUC')
    axes[0, 1].set_title('AUC')
    axes[0, 1].legend()
    
    # Plot F1
    axes[0, 2].plot(metrics['val_f1s'], label='Val F1')
    axes[0, 2].set_title('F1 Score')
    axes[0, 2].legend()
    
    # Plot Precision
    axes[1, 0].plot(metrics['val_precisions'], label='Val Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].legend()
    
    # Plot Recall
    axes[1, 1].plot(metrics['val_recalls'], label='Val Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].legend()
    
    # Plot Accuracy
    axes[1, 2].plot(metrics['val_accuracies'], label='Val Accuracy')
    axes[1, 2].set_title('Accuracy')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plot_filename = f'{model_type}_training_metrics_{timestamp}.png'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plots saved to: {plot_path}")

def main():
    import argparse
    import os
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified Training Script for RETAIN and TFCAM')
    parser.add_argument('--model', type=str, required=True, help='Model type: retain, tfcam, hcta, enhanced_tfcam, mstca, ctga')
    parser.add_argument('--hyperparameter-search', action='store_true', 
                       help='Enable hyperparameter search')
    parser.add_argument('--n-combinations', type=int, default=10,
                       help='Number of hyperparameter combinations to try')
    parser.add_argument('--config', type=str, required=False,
                       help='Path to configuration YAML file (if not provided, will use config/{model}_config.yaml)')
    args = parser.parse_args()

    # Auto-determine config if not provided
    if args.config is None:
        args.config = f"config/{args.model.lower()}_config.yaml"
        print(f"[INFO] Using config: {args.config}")

    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    # Set model type in config
    config['model']['type'] = args.model.lower()

    # Check if best hyperparameters exist
    best_params_path = os.path.join(config['paths']['results_path'], 'best_hyperparameters.yaml')
    if os.path.exists(best_params_path) and not args.hyperparameter_search:
        best_params = load_best_params(best_params_path)
        config['model'].update(best_params)
        print("✅ Loaded best hyperparameters.")
    else:
        print(f"⚠️ Best hyperparameters file not found or hyperparameter search enabled. Using default parameters from {config_path}.")

    config = update_config_with_timestamp(config)
    config['model']['device'] = get_best_device()
    print(f"Using device: {config['model']['device']}")

    # Prepare data
    train_data, val_data, test_data = prepare_data(config)
    print("✅ Data prepared.")
    
    best_model_path = None
    if args.hyperparameter_search:
        # Perform hyperparameter search
        print(f"🔍 Starting hyperparameter search with {args.n_combinations} combinations...")
        best_params, best_f1, best_auc, best_model_path = hyperparameter_search(
            config, train_data, val_data, test_data, 
            n_combinations=args.n_combinations
        )
        
        if best_params:
            # Update config with best parameters
            config['model'].update(best_params)
            print(f"✅ Best F1: {best_f1:.4f}")
            print(f"✅ Best AUC: {best_auc:.4f}")
            print(f"✅ Best params: {best_params}")
            print(f"✅ Best model saved at: {best_model_path}")
            save_best_params(best_params, config, config['timestamp'])
        return  # 只做超参数搜索和保存，不再重复训练

    # Create model with final parameters
    model = create_model(config)
    print("✅ Model created.")
    
    # Set up training
    device = torch.device(config['model']['device'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(train_data, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['model']['batch_size'])
    
    # Train model
    metrics = train_model(model, train_loader, val_loader, criterion, optimizer, config)
    print("✅ Model trained.")
    
    # Save training results
    save_training_results(metrics, config)
    plot_training_metrics(metrics, config)
    
    # Print final results
    best_f1 = max(metrics['val_f1s'])
    best_auc = max(metrics['val_aucs'])
    print(f"\n🎉 Training completed!")
    print(f"📊 Best F1 Score: {best_f1:.4f}")
    print(f"📊 Best AUC Score: {best_auc:.4f}")

if __name__ == '__main__':
    main() 