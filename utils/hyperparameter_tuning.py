import os
import yaml
import itertools
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from models.retain import RETAIN
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data(config):
    data = pd.read_csv(config['data']['processed_path']).fillna(0)
    input_dim = config['model']['input_dim']
    num_period = config['data']['month_count'] // 3
    features = data.drop(columns=['TMA_Acct', 'ESRD']).values
    labels = data['ESRD'].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = features.reshape(len(features), num_period, input_dim)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        features_tensor, labels_tensor, test_size=0.4, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    X_train_np = X_train.numpy().reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_np, y_train.numpy())
    X_train_resampled = X_train_resampled.reshape(-1, num_period, input_dim)

    train_data = TensorDataset(
        torch.tensor(X_train_resampled, dtype=torch.float32),
        torch.tensor(y_train_resampled, dtype=torch.float32)
    )
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    return train_data, val_data, test_data

def evaluate_model(model, loader, criterion, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted_probs = torch.sigmoid(outputs).view(-1)
            y_pred.extend(predicted_probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return f1_score(y_true, np.array(y_pred) > 0.5)

def hyperparameter_search(config, train_data, val_data):
    hyperparams = config['hyperparameters']
    combinations = list(itertools.product(*hyperparams.values()))
    random.shuffle(combinations)

    device = torch.device(config['model']['device'])
    save_path = config['paths']['save_path']
    os.makedirs(save_path, exist_ok=True)

    best_f1, best_params = -np.inf, None
    criterion = nn.BCEWithLogitsLoss()

    for params in tqdm(combinations, desc="Hyperparameter Tuning"):
        params_dict = dict(zip(hyperparams.keys(), params))

        model = RETAIN(
            input_dim=config['model']['input_dim'],
            emb_dim=params_dict['emb_dim'],
            hidden_dim=params_dict['hidden_dim'],
            output_dim=config['model']['output_dim'],
            dropout=params_dict['dropout']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params_dict['learning_rate'])

        train_loader = DataLoader(train_data, batch_size=params_dict['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=params_dict['batch_size'])

        for epoch in range(config['tuning']['n_epochs']):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

        val_f1 = evaluate_model(model, val_loader, criterion, device)

        if val_f1 > best_f1:
            best_f1, best_params = val_f1, params_dict
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model_hyperparams.pt'))
            print(f"New Best Params: {best_params}, F1: {best_f1:.4f}")

    return best_params, best_f1

def save_best_params(best_params, config):
    save_path = os.path.join(config['paths']['results_path'], 'best_hyperparameters.yaml')
    os.makedirs(config['paths']['results_path'], exist_ok=True)
    with open(save_path, 'w') as file:
        yaml.dump(best_params, file)
    print(f"Best hyperparameters saved to {save_path}")

def main():
    config = load_config('../config/config.yaml')
    train_data, val_data, _ = prepare_data(config)
    best_params, best_f1 = hyperparameter_search(config, train_data, val_data)
    save_best_params(best_params, config)
    print(f"Hyperparameter tuning complete. Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
