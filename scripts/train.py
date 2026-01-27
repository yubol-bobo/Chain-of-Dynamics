#!/usr/bin/env python3
"""
Unified Training Script for Clinical Models

This script provides a unified training interface for BiLSTM, RETAIN, CoI, Transformer,
AdaCare, and StageNet models, with support for hyperparameter tuning,
class-weighted losses, and comprehensive training metrics.
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
from collections import Counter
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Import enhanced preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.data.enhanced_preprocessing import EnhancedTemporalPreprocessor
    ENHANCED_PREPROCESSING_AVAILABLE = True
    print("[INFO] Enhanced preprocessing available")
except ImportError as e:
    ENHANCED_PREPROCESSING_AVAILABLE = False
    print(f"[WARNING] Enhanced preprocessing not available: {e}")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.retain import RETAIN
from src.models.coi import CoI
from src.models.bilstm import BiLSTM
from src.models.transformer import TransformerModel
from src.models.adacare import AdaCare
from src.models.stagenet import StageNet

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

def set_random_seeds(seed=2025):
    """Set random seeds for reproducible sampling and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(log_path):
    """Create a file-only logger for training output."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

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
    print(f"Generated timestamp: {timestamp}")
    return config

def get_best_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA available! Using GPU: {device_name}")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS available! Using Apple Silicon GPU")
        return 'mps'
    else:
        print("GPU not available. Using CPU")
        return 'cpu'

def prepare_data(config, use_enhanced=True):
    """
    Prepare data for training with optional enhanced preprocessing.

    Args:
        config: Configuration dictionary
        use_enhanced: If True, use enhanced temporal-aware preprocessing

    Returns:
        train_data, val_data, test_data, input_dim: TensorDataset objects and input dimension
    """
    print("Preparing data...")

    # Check if we should use enhanced preprocessing
    if use_enhanced and ENHANCED_PREPROCESSING_AVAILABLE:
        return prepare_data_enhanced(config)
    else:
        if use_enhanced and not ENHANCED_PREPROCESSING_AVAILABLE:
            print("[WARNING] Enhanced preprocessing requested but not available, using standard preprocessing")
        return prepare_data_standard(config)

def prepare_data_enhanced(config):
    """Prepare data using enhanced temporal-aware preprocessing."""
    print("[INFO] Using ENHANCED preprocessing pipeline...")

    processed_path = config['data']['processed_path']

    # For CKD dataset (CSV format)
    if processed_path.endswith('.csv'):
        preprocessor = EnhancedTemporalPreprocessor()
        train_data, val_data, test_data, input_dim, class_weights = preprocessor.preprocess_ckd_data(processed_path)

        # Store class weights in config for use in training
        config['enhanced_class_weights'] = class_weights

        return train_data, val_data, test_data, input_dim

    # For MIMIC dataset (numpy format)
    else:
        print("[INFO] Using enhanced MIMIC preprocessing...")
        preprocessor = EnhancedTemporalPreprocessor()
        label_path = config['data']['label_path']
        train_data, val_data, test_data, input_dim, class_weights = preprocessor.preprocess_mimic_data(processed_path, label_path)

        # Store class weights in config for use in training
        config['enhanced_class_weights'] = class_weights

        return train_data, val_data, test_data, input_dim

def prepare_data_standard(config):
    """Prepare data using standard preprocessing (existing logic)."""
    print("[INFO] Using STANDARD preprocessing pipeline...")
    processed_path = config['data']['processed_path']
    label_path = config['data'].get('label_path')
    if processed_path.endswith('.npy'):
        features = np.load(processed_path)
        if label_path and label_path.endswith('.npy'):
            labels = np.load(label_path)
        else:
            raise ValueError('Label path must be provided and end with .npy when using .npy features.')
        # Check for all-NaN features before imputation
        all_nan_features = np.all(np.isnan(features), axis=(0, 1))
        if np.any(all_nan_features):
            drop_indices = np.where(all_nan_features)[0]
            print("[INFO] Dropping all-NaN features at indices:", drop_indices)
            features = features[..., ~all_nan_features]
        # Split the dataset into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.4, stratify=labels, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        # MICE Imputation and Normalization (fit only on train, transform all)
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        n_patients_train, timesteps, n_features = X_train.shape
        n_patients_val, _, _ = X_val.shape
        n_patients_test, _, _ = X_test.shape
        X_train_2d = X_train.reshape(-1, n_features)
        X_val_2d = X_val.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        print('[DEBUG] Before imputation: Any NaNs in train?', np.isnan(X_train_2d).any())
        imputer = IterativeImputer(random_state=42, max_iter=10)
        X_train_imputed = imputer.fit_transform(X_train_2d)
        X_val_imputed = imputer.transform(X_val_2d)
        X_test_imputed = imputer.transform(X_test_2d)
        print('[DEBUG] After imputation: Any NaNs in train?', np.isnan(X_train_imputed).any())
        print('[DEBUG] After imputation: Any NaNs in val?', np.isnan(X_val_imputed).any())
        print('[DEBUG] After imputation: Any NaNs in test?', np.isnan(X_test_imputed).any())
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        print('[DEBUG] After normalization: Any NaNs in train?', np.isnan(X_train_scaled).any())
        print('[DEBUG] After normalization: Any NaNs in val?', np.isnan(X_val_scaled).any())
        print('[DEBUG] After normalization: Any NaNs in test?', np.isnan(X_test_scaled).any())
        X_train = X_train_scaled.reshape(n_patients_train, timesteps, n_features)
        X_val = X_val_scaled.reshape(n_patients_val, timesteps, n_features)
        X_test = X_test_scaled.reshape(n_patients_test, timesteps, n_features)
        # Forcibly replace any remaining NaNs/Infs
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        print('[DEBUG] Feature min/max after all preprocessing (train):', np.nanmin(X_train), np.nanmax(X_train))
        print('[DEBUG] Feature min/max after all preprocessing (val):', np.nanmin(X_val), np.nanmax(X_val))
        print('[DEBUG] Feature min/max after all preprocessing (test):', np.nanmin(X_test), np.nanmax(X_test))
        # Save imputed/scaled test set for analysis
        model_type = config['model'].get('type', 'model')
        results_path = config['paths']['results_path']
        np.save(os.path.join(results_path, f"{model_type}_X_test_imputed.npy"), X_test)
        np.save(os.path.join(results_path, f"{model_type}_y_test_imputed.npy"), y_test)
        # Compute class weights from the original training split (no resampling).
        y_train_flat = y_train if y_train.ndim == 1 else y_train.argmax(axis=1)
        y_train_flat = np.asarray(y_train_flat, dtype=np.int64)
        class_counts = Counter(y_train_flat.tolist())
        total = sum(class_counts.values())
        class_weights = {cls: total / count for cls, count in class_counts.items()}
        neg_count = class_counts.get(0, 0)
        pos_count = class_counts.get(1, 0)
        if pos_count > 0 and neg_count > 0:
            config['pos_weight'] = float(neg_count / pos_count)
        max_weight = max(class_weights.values())
        class_weights = {cls: w / max_weight for cls, w in class_weights.items()}
        print(f"[INFO] Using class weights in loss: {class_weights}")
        class_weights_tensor = torch.tensor(
            [class_weights[cls] for cls in sorted(class_weights.keys())], dtype=torch.float32
        )
        config['class_weights_tensor'] = class_weights_tensor
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
        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)
        test_data = TensorDataset(X_test, y_test)

        # Calculate input_dim from the tensor shape
        input_dim = X_train.shape[2] if len(X_train.shape) == 3 else config['model']['input_dim']
        return train_data, val_data, test_data, input_dim
    else:
        data = pd.read_csv(processed_path)
        shape_0 = data.shape[0]
        print(f"Data loaded: {data.shape}")

        num_period = config['data']['month_count'] // 3

        # Extract features and labels without early fillna so we can impute temporally.
        features_df = data.drop(columns=['TMA_Acct', 'ESRD'])
        labels_np = data['ESRD'].to_numpy(dtype=np.float32)
        print(f"Features shape: {features_df.shape}, Labels shape: {labels_np.shape}")

        # Calculate the correct input_dim based on actual data
        total_features = features_df.shape[1]
        input_dim = total_features // num_period

        # Ensure the dimensions work out correctly
        expected_total = input_dim * num_period
        if total_features != expected_total:
            print(f"[WARNING] Feature count mismatch: {total_features} != {input_dim} * {num_period} = {expected_total}")
            if total_features > expected_total:
                features_df = features_df.iloc[:, :expected_total]
                print(f"[INFO] Trimmed features to {features_df.shape[1]} to match expected dimensions")
            else:
                padding_needed = expected_total - total_features
                padding_cols = [f"padding_{i}" for i in range(padding_needed)]
                padding_df = pd.DataFrame(
                    np.zeros((features_df.shape[0], padding_needed), dtype=np.float32),
                    columns=padding_cols,
                )
                features_df = pd.concat([features_df, padding_df], axis=1)
                print(f"[INFO] Padded features with {padding_needed} zero columns")

        print(f"Final input dim: {input_dim}, Num periods: {num_period}")

        # Reshape raw features to [n_patients, timesteps, input_dim]
        features_np = features_df.to_numpy(dtype=np.float32, copy=True)
        features_reshaped = features_np.reshape(shape_0, num_period, input_dim)

        def temporal_fill_per_patient(x_3d):
            """Forward/backward fill along time for each patient."""
            filled = x_3d.copy()
            for i in range(filled.shape[0]):
                filled[i] = pd.DataFrame(filled[i]).ffill().bfill().to_numpy(dtype=np.float32)
            return filled

        # Temporal imputation (no engineered features; preserves 36 features).
        features_temporal_filled = temporal_fill_per_patient(features_reshaped)

        # Split before median imputation/scaling to avoid leakage.
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_temporal_filled, labels_np, test_size=0.4, stratify=labels_np, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        # Median imputation using train statistics only (flatten time dimension).
        train_flat = X_train.reshape(-1, input_dim)
        feature_medians = np.nanmedian(train_flat, axis=0)
        feature_medians = np.nan_to_num(feature_medians, nan=0.0, posinf=0.0, neginf=0.0)

        def apply_feature_medians(x_3d, medians):
            imputed = x_3d.copy()
            for j in range(imputed.shape[2]):
                mask = np.isnan(imputed[:, :, j])
                if np.any(mask):
                    imputed[:, :, j][mask] = medians[j]
            return imputed

        X_train = apply_feature_medians(X_train, feature_medians)
        X_val = apply_feature_medians(X_val, feature_medians)
        X_test = apply_feature_medians(X_test, feature_medians)

        # Standard scaling using train statistics only.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, input_dim)).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, input_dim)).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, input_dim)).reshape(X_test.shape)

        # Final cleanup for any remaining NaN/Inf.
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Save imputed/scaled test set for analysis.
        model_type = config['model'].get('type', 'model')
        results_path = config['paths']['results_path']
        np.save(os.path.join(results_path, f"{model_type}_X_test_imputed.npy"), X_test_scaled)
        np.save(os.path.join(results_path, f"{model_type}_y_test_imputed.npy"), y_test)

        def to_tensor(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return torch.as_tensor(x, dtype=torch.float32)

        X_train_tensor = to_tensor(X_train_scaled)
        y_train_tensor = to_tensor(y_train)
        X_val_tensor = to_tensor(X_val_scaled)
        y_val_tensor = to_tensor(y_val)
        X_test_tensor = to_tensor(X_test_scaled)
        y_test_tensor = to_tensor(y_test)

        print(
            "Data split - Train: {}, Val: {}, Test: {}".format(
                get_shape(X_train_tensor), get_shape(X_val_tensor), get_shape(X_test_tensor)
            )
        )

        # Create Tensor datasets
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        val_data = TensorDataset(X_val_tensor, y_val_tensor)
        test_data = TensorDataset(X_test_tensor, y_test_tensor)

        # Compute class weights from the original training split.
        y_train_flat_np = y_train if np.ndim(y_train) == 1 else y_train.argmax(axis=1)
        y_train_flat_np = np.asarray(y_train_flat_np, dtype=np.int64)
        class_counts = Counter(y_train_flat_np.tolist())
        total = sum(class_counts.values())
        class_weights = {cls: total / count for cls, count in class_counts.items()}
        neg_count = class_counts.get(0, 0)
        pos_count = class_counts.get(1, 0)
        if pos_count > 0 and neg_count > 0:
            config['pos_weight'] = float(neg_count / pos_count)
        max_weight = max(class_weights.values())
        class_weights = {cls: w / max_weight for cls, w in class_weights.items()}
        print(f"[INFO] Using class weights in loss: {class_weights}")
        class_weights_tensor = torch.tensor(
            [class_weights[cls] for cls in sorted(class_weights.keys())], dtype=torch.float32
        )
        config['class_weights_tensor'] = class_weights_tensor

        return train_data, val_data, test_data, input_dim

def create_model(config, input_dim=None):
    """Create model based on config type"""
    model_type = config['model'].get('type', 'retain')

    # Use provided input_dim or fall back to config
    actual_input_dim = input_dim if input_dim is not None else config['model']['input_dim']

    if model_type.lower() == 'coi':
        model = CoI(
            input_dim=actual_input_dim,
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
    elif model_type.lower() == 'bilstm':
        model = BiLSTM(
            input_dim=actual_input_dim,
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model'].get('num_layers', 2),
            output_dim=config['model']['output_dim'],
            dropout=config['model'].get('dropout', 0.2)
        )
    elif model_type.lower() == 'retain':
        model = RETAIN(
            input_dim=actual_input_dim,
            emb_dim=config['model']['emb_dim'],
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            dropout=config['model'].get('dropout', 0.2)
        )
    elif model_type.lower() == 'transformer':
        model = TransformerModel(
            input_dim=actual_input_dim,
            emb_dim=config['model']['emb_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            output_dim=config['model']['output_dim'],
            dropout=config['model'].get('dropout', 0.2),
            max_seq_len=config['model'].get('max_seq_len', 50)
        )
    elif model_type.lower() == 'adacare':
        model = AdaCare(
            input_dim=actual_input_dim,
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            num_heads=config['model'].get('num_heads', 2),
            dropout=config['model'].get('dropout', 0.2),
            calibration_dim=config['model'].get('calibration_dim', 64)
        )
    elif model_type.lower() == 'stagenet':
        model = StageNet(
            input_dim=actual_input_dim,
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            num_stages=config['model'].get('num_stages', 4),
            kernel_size=config['model'].get('kernel_size', 3),
            dropout=config['model'].get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: retain, coi, bilstm, transformer, adacare, stagenet")
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, config, log_alpha=False, logger=None, disable_tqdm=False):
    """
    Train model with early stopping and comprehensive metrics.
    Early stopping is now based on validation loss instead of F1 score.
    """
    log_info = logger.info if logger else print
    log_warn = logger.warning if logger else print
    log_info("Training model...")
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

    # Use class weights if available (standard or enhanced)
    if 'enhanced_class_weights' in config:
        # Enhanced preprocessing class weights
        class_weights = config['enhanced_class_weights']
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        log_info(f"[INFO] Using enhanced class weights - pos_weight: {pos_weight.item():.4f}")
    elif config['model'].get('output_dim', 1) == 1 and 'pos_weight' in config:
        pos_weight = torch.tensor([config['pos_weight']], dtype=torch.float32).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        log_info(f"[INFO] Using standard pos_weight: {pos_weight.item():.4f}")
    elif 'class_weights_tensor' in config:
        # Standard preprocessing class weights
        class_weights_tensor = config['class_weights_tensor'].to(next(model.parameters()).device)
        if config['model'].get('output_dim', 1) == 1:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        log_info(f"[INFO] Using standard class weights")
    else:
        # No class weights
        criterion = torch.nn.BCEWithLogitsLoss()
        log_info(f"[INFO] Using unweighted BCE loss")

    for epoch in tqdm(range(n_epochs), desc="Epochs", disable=disable_tqdm, leave=False):
        model.train()
        epoch_losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Check for NaN in model outputs
            if torch.isnan(outputs).any():
                log_warn(f"[WARNING] NaN detected in model outputs at epoch {epoch}, skipping batch")
                continue

            loss = criterion(outputs, labels.unsqueeze(1))

            # Check for NaN in loss
            if torch.isnan(loss):
                log_warn(f"[WARNING] NaN detected in loss at epoch {epoch}, skipping batch")
                continue

            loss.backward()

            # Check for NaN in gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break

            if has_nan_grad:
                log_warn(f"[WARNING] NaN detected in gradients at epoch {epoch}, skipping batch")
                optimizer.zero_grad()
                continue

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                    log_warn("[DEBUG] First validation batch input (sample): " + str(inputs.flatten()[:10].cpu().numpy()))
                    log_warn("[DEBUG] First validation batch label (sample): " + str(labels.flatten()[:10].cpu().numpy()))
                    log_warn("[DEBUG] First validation batch model output (pre-sigmoid, sample): " + str(outputs.flatten()[:10].cpu().numpy()))

        binary_preds = (np.array(y_pred) > 0.5).astype(int)
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        if np.isnan(y_true_np).any() or np.isnan(y_pred_np).any():
            log_warn("[WARNING] NaN detected in validation labels or predictions!")
            log_warn(f"y_true sample: {y_true_np[:10]}")
            log_warn(f"y_pred sample: {y_pred_np[:10]}")
            log_warn("Skipping metric calculation for this epoch.")
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

        # Optional: Log DyT alpha values for CoI
        if log_alpha and model_type.lower() == 'coi':
            alpha_values = []
            for module in model.modules():
                if hasattr(module, 'alpha') and isinstance(module.alpha, torch.nn.Parameter):
                    alpha_values.append(module.alpha.item())
            if alpha_values:
                log_info(f"DyT alpha values: {alpha_values}")

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
                log_info("Early stopping triggered (validation loss not improving).")
                break
                
    return metrics

def hyperparameter_search(config, train_data, val_data, test_data, actual_input_dim, n_combinations=10, logger=None):
    model_type = config['model'].get('type', 'retain')
    device = torch.device(config['model']['device'])
    save_path = config['paths']['save_path']
    log_info = logger.info if logger else print
    # train_model writes the best-epoch checkpoint to this path; clear it per combo.
    best_epoch_ckpt_path = os.path.join(
        save_path, f"{model_type}_best_model_{config.get('timestamp', 'unknown')}.pt"
    )
    if model_type.lower() == 'coi':
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
    elif model_type.lower() == 'bilstm':
        hyperparameters = {
            'hidden_dim': [32, 64, 128],
            'num_layers': [1, 2, 3],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [64, 128],
            'dropout': [0, 0.2, 0.4]
        }
    elif model_type.lower() == 'retain':
        hyperparameters = {
            'emb_dim': [16, 32, 64],
            'hidden_dim': [32, 64, 128],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [64, 128],
            'dropout': [0, 0.2, 0.4]
        }
    elif model_type.lower() == 'transformer':
        hyperparameters = {
            'emb_dim': [64, 128, 256],
            'hidden_dim': [256, 512, 1024],
            'num_heads': [4, 8, 16],
            'num_layers': [2, 4, 6],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [32, 64, 128],
            'dropout': [0.1, 0.2, 0.3]
        }
    elif model_type.lower() == 'adacare':
        hyperparameters = {
            'hidden_dim': [128, 256, 512],
            'num_heads': [1, 2, 4],
            'calibration_dim': [32, 64, 128],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [32, 64, 128],
            'dropout': [0.1, 0.2, 0.3]
        }
    elif model_type.lower() == 'stagenet':
        hyperparameters = {
            'hidden_dim': [128, 256, 512],
            'num_stages': [3, 4, 5],
            'kernel_size': [3, 5, 7],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [32, 64, 128],
            'dropout': [0.1, 0.2, 0.3]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: retain, coi, bilstm, transformer, adacare, stagenet")
    param_names = list(hyperparameters.keys())
    param_values = list(hyperparameters.values())
    all_combinations = list(itertools.product(*param_values))
    if len(all_combinations) > n_combinations:
        all_combinations = random.sample(all_combinations, n_combinations)
    best_val_loss, best_f1, best_auc, best_hyperparams, best_model_path = float('inf'), -np.inf, -np.inf, None, None
    criterion = nn.BCEWithLogitsLoss()
    log_info(f"Starting hyperparameter search for {model_type.upper()}...")
    log_info(f"Testing {len(all_combinations)} combinations (full training each)...")
    pbar = tqdm(all_combinations, desc=f"{model_type.upper()} combinations", leave=True)
    for params in pbar:
        params_dict = dict(zip(param_names, params))
        config['model'].update(params_dict)
        # Avoid reusing a stale best-epoch checkpoint from a prior combination.
        if os.path.exists(best_epoch_ckpt_path):
            try:
                os.remove(best_epoch_ckpt_path)
            except OSError:
                pass
        model = create_model(config, actual_input_dim)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=params_dict['learning_rate'])
        train_loader = DataLoader(train_data, batch_size=params_dict['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=params_dict['batch_size'])
        metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, config,
            logger=logger, disable_tqdm=True
        )
        val_losses = metrics['val_losses']
        best_epoch = int(np.argmin(val_losses))
        current_val_loss = val_losses[best_epoch]
        current_f1 = metrics['val_f1s'][best_epoch]
        current_auc = metrics['val_aucs'][best_epoch]
        # train_model saves the best-epoch checkpoint to this timestamped path.
        if os.path.exists(best_epoch_ckpt_path):
            best_epoch_state = torch.load(best_epoch_ckpt_path, map_location=device)
            best_epoch_state_dict = best_epoch_state.get('model_state_dict', model.state_dict())
        else:
            best_epoch_state_dict = model.state_dict()
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
                'model_state_dict': best_epoch_state_dict,
                'hyperparams': config['model']
            }, candidate_model_path)
            log_info(
                "New best model (Val Loss: %.4f, F1: %.4f, AUC: %.4f) with params: %s"
                % (best_val_loss, best_f1, best_auc, params_dict)
            )
        pbar.set_postfix(best_val_loss=f"{best_val_loss:.4f}")
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
    log_info(f"Hyperparameter search summary saved to: {summary_path}")
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
    
    print(f"Training metrics saved to: {metrics_path}")

def save_best_params(best_params, config, timestamp):
    """Save best hyperparameters"""
    results_path = config['paths']['results_path']
    best_params_path = os.path.join(results_path, f'best_hyperparameters_{timestamp}.yaml')
    
    with open(best_params_path, 'w') as f:
        yaml.dump(best_params, f)
    
    print(f"Best hyperparameters saved to: {best_params_path}")

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
    
    print(f"Training plots saved to: {plot_path}")

def main():
    import argparse
    import os
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified Training Script for Clinical Models')
    parser.add_argument('--model', type=str, required=True, help='Model type: retain, coi, bilstm, transformer, adacare, stagenet')
    parser.add_argument('--dataset', type=str, choices=['ckd', 'mimic'], default='ckd',
                       help='Dataset to use: ckd or mimic (default: ckd)')
    parser.add_argument('--hyperparameter-search', action='store_true',
                       help='Enable hyperparameter search')
    parser.add_argument('--n-combinations', type=int, default=300,
                       help='Number of hyperparameter combinations to try')
    parser.add_argument('--config', type=str, required=False,
                       help='Path to configuration YAML file (overrides --dataset flag)')
    parser.add_argument('--enhanced', action='store_true', default=True,
                       help='Use enhanced temporal-aware preprocessing (default: True)')
    parser.add_argument('--standard', action='store_true',
                       help='Use standard preprocessing (overrides --enhanced)')
    args = parser.parse_args()
    set_random_seeds(2025)
    print("[INFO] Random seed set to 2025")

    # Auto-determine config if not provided
    if args.config is None:
        if args.dataset == 'mimic':
            args.config = f"config/mimiciv_{args.model.lower()}_config.yaml"
        else:
            args.config = f"config/{args.model.lower()}_config.yaml"
        print(f"[INFO] Using config: {args.config} for {args.dataset.upper()} dataset")

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
        print("Loaded best hyperparameters.")
    else:
        print(f"Best hyperparameters file not found or hyperparameter search enabled. Using default parameters from {config_path}.")

    config = update_config_with_timestamp(config)
    config['model']['device'] = get_best_device()
    print(f"Using device: {config['model']['device']}")
    os.makedirs(config['paths']['results_path'], exist_ok=True)
    log_path = os.path.join(
        config['paths']['results_path'],
        f"{config['model']['type']}_{config['timestamp']}_train.log",
    )
    logger = setup_logger(log_path)
    logger.info("Random seed: 2025")
    logger.info("Config path: %s", config_path)
    logger.info("Device: %s", config['model']['device'])
    print(f"Training log: {log_path}")
    stdout_original = sys.stdout
    log_file = open(log_path, mode="a", encoding="utf-8")
    sys.stdout = log_file

    # Determine preprocessing method
    use_enhanced = args.enhanced and not args.standard
    if args.standard:
        print("[INFO] Standard preprocessing requested via --standard flag")
        use_enhanced = False
    elif args.enhanced:
        print("[INFO] Enhanced preprocessing enabled (default)")
    else:
        print("[INFO] Enhanced preprocessing disabled via --no-enhanced flag")
        use_enhanced = False

    # Prepare data
    train_data, val_data, test_data, actual_input_dim = prepare_data(config, use_enhanced=use_enhanced)
    print("Data prepared.")
    
    best_model_path = None
    if args.hyperparameter_search:
        # Perform hyperparameter search
        print(f"Starting hyperparameter search with {args.n_combinations} combinations...")
        best_params, best_f1, best_auc, best_model_path = hyperparameter_search(
            config, train_data, val_data, test_data, actual_input_dim,
            n_combinations=args.n_combinations, logger=logger
        )

        if best_params:
            # Update config with best parameters
            config['model'].update(best_params)
            logger.info("Best F1: %.4f", best_f1)
            logger.info("Best AUC: %.4f", best_auc)
            logger.info("Best params: %s", best_params)
            logger.info("Best model saved at: %s", best_model_path)
            print(f"Best model saved at: {best_model_path}")
            save_best_params(best_params, config, config['timestamp'])
    else:
        # Train with default parameters
        print("Training with default parameters from config...")

        # Create model with default parameters
        model = create_model(config, actual_input_dim)
        print("[SUCCESS] Model created.")

        # Set up training
        device = torch.device(config['model']['device'])
        model.to(device)

        # Use enhanced class weights if available
        if hasattr(train_data.dataset, 'class_weights') if hasattr(train_data, 'dataset') else False:
            class_weights = train_data.dataset.class_weights
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using weighted BCE loss with pos_weight: {pos_weight.item():.4f}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("Using standard BCE loss")

        optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])

        train_loader = DataLoader(train_data, batch_size=config['model']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['model']['batch_size'])

        # Train model
        metrics = train_model(model, train_loader, val_loader, criterion, optimizer, config, logger=logger)
        print("[SUCCESS] Model trained.")

        # Save the best model
        best_model_path = os.path.join(config['paths']['save_path'], f"{config['model']['type']}_{config['timestamp']}_best.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparams': config['model'],
            'metrics': metrics
        }, best_model_path)
        print(f"Best model saved at: {best_model_path}")
    
    # # Print final results
    # best_f1 = max(metrics['val_f1s'])
    # best_auc = max(metrics['val_aucs'])
    # print(f"\n🎉 Training completed!")
    # print(f"📊 Best F1 Score: {best_f1:.4f}")
    # print(f"📊 Best AUC Score: {best_auc:.4f}")

    # After hyperparameter search and best model selection, evaluate on test set
    print("\n[INFO] Evaluating best model on test set...")
    # Load best model
    device = torch.device(config['model'].get('device', 'cpu'))
    model = create_model(config, actual_input_dim)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    test_loader = DataLoader(test_data, batch_size=config['model']['batch_size'], shuffle=False)
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                pos_probs = probs
            else:
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)
                pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs.max(axis=1)
            all_preds.append(preds)
            all_probs.append(pos_probs)
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    f1 = f1_score(all_labels, all_preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auroc = float('nan')
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    print(f"[TEST] F1: {f1:.4f}, AUROC: {auroc:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}")
    # Save metrics to YAML
    metrics = {
        'f1': float(f1),
        'auroc': float(auroc),
        'accuracy': float(acc),
        'precision': float(prec)
    }
    model_type = config['model'].get('type', 'model')
    metrics_path = os.path.join(config['paths']['results_path'], f'{model_type}_test_metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    print(f"[TEST] Metrics saved to: {metrics_path}")
    sys.stdout = stdout_original
    log_file.close()
    print(f"Best model saved at: {best_model_path}")
    print(f"Test metrics saved to: {metrics_path}")
    print(f"Training log: {log_path}")

if __name__ == '__main__':
    main() 
