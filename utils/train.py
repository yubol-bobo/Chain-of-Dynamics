import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
from models.retain import RETAIN
# from utils.metrics import create_metrics_dataframe
# from utils.visualization import plot_metrics

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

def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    device = torch.device(config['model']['device'])
    model.to(device)
    best_f1, trigger_times = -np.inf, 0
    patience, n_epochs = config['model']['patience'], config['model']['n_epochs']
    save_path = config['paths']['save_path']

    metrics = {
        'train_losses': [], 'val_losses': [], 'val_aucs': [],
        'val_recalls': [], 'val_accuracies': [], 'val_precisions': [], 'val_f1s': []
    }

    for epoch in range(n_epochs):
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss_epoch.append(criterion(outputs, labels.unsqueeze(1)).item())
                predicted_probs = torch.sigmoid(outputs).view(-1)
                y_pred.extend(predicted_probs.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        binary_preds = (np.array(y_pred) > 0.5).astype(int)
        val_f1 = f1_score(y_true, binary_preds)
        metrics['val_losses'].append(np.mean(val_loss_epoch))
        metrics['val_aucs'].append(roc_auc_score(y_true, y_pred))
        metrics['val_recalls'].append(recall_score(y_true, binary_preds))
        metrics['val_accuracies'].append(accuracy_score(y_true, binary_preds))
        metrics['val_precisions'].append(precision_score(y_true, binary_preds))
        metrics['val_f1s'].append(val_f1)

        print(f"Epoch [{epoch+1}/{n_epochs}] Train Loss: {metrics['train_losses'][-1]:.4f} "
              f"Val Loss: {metrics['val_losses'][-1]:.4f} Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(save_path, 'retain_best_model.pt'))
            print(f"Best model updated with F1: {best_f1:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break
    return metrics

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))

    config_path = os.path.join(root_dir, 'config', 'config.yaml')
    best_params_path = os.path.join(root_dir, 'Outputs', 'best_hyperparameters.yaml')

    config = load_config(config_path)

    # Check if best hyperparameters exist
    if os.path.exists(best_params_path):
        best_params = load_best_params(best_params_path)
        config['model'].update(best_params)
        print("✅ Loaded best hyperparameters.")
    else:
        print("⚠️ Best hyperparameters file not found. Using default parameters from config.yaml.")

    device = torch.device(config['model']['device'])

    train_data, val_data, _ = prepare_data(config)
    model = RETAIN(
        input_dim=config['model']['input_dim'],
        emb_dim=config['model']['emb_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_data, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['model']['batch_size'])

    train_model(model, train_loader, val_loader, criterion, optimizer, config, device)

if __name__ == '__main__':
    main()
