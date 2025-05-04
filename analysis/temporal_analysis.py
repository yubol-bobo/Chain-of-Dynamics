import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_all_temporal(model, data_loader, device):
    model.eval()
    temporal_list = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            temporal_batch = model.alpha_weights.squeeze(-1).cpu().numpy()
            temporal_list.append(temporal_batch)
    
    return np.concatenate(temporal_list, axis=0)

def plot_global_temporal_attention(model, data_loader, device, plot_type='bar', save_dir="./visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    
    temporal_weights = get_all_temporal(model, data_loader, device)
    time_steps = [f"t{i}" for i in range(temporal_weights.shape[1])]

    mean_temporal = temporal_weights.mean(axis=0)

    plt.figure(figsize=(12, 8))
    
    if plot_type == 'bar':
        sns.barplot(x=time_steps, y=mean_temporal, palette='viridis')
        plt.xlabel('Time Step')
        plt.ylabel('Mean Attention Weight')

    plt.title('Global Temporal Attention')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"global_temporal_attention_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_temporal_comparison(model, data_loader, device, save_dir="./visualizations"):
    os.makedirs(save_dir, exist_ok=True)

    temporal_weights, labels = [], []
    
    with torch.no_grad():
        for inputs, lbl in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            temporal_batch = model.alpha_weights.squeeze(-1).cpu().numpy()
            temporal_weights.append(temporal_batch)
            labels.append(lbl.numpy())
    
    temporal_weights = np.concatenate(temporal_weights, axis=0)
    labels = np.concatenate(labels, axis=0)

    pos_mean = temporal_weights[labels == 1].mean(axis=0)
    neg_mean = temporal_weights[labels == 0].mean(axis=0)
    time_steps = [f"t{i}" for i in range(temporal_weights.shape[1])]

    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, pos_mean, 'r-o', label='Positive Outcome')
    plt.plot(time_steps, neg_mean, 'b-o', label='Negative Outcome')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Attention Weight')
    plt.title('Temporal Attention Comparison')
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"temporal_comparison_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
