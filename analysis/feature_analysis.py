import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_feature_contributions(model, data_loader, device):
    model.eval()
    contributions_list = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)

            batch_contrib = []
            for idx in range(inputs.size(0)):
                contrib = model.compute_contributions(patient_idx=idx)
                batch_contrib.append(contrib)

            contributions_list.append(np.stack(batch_contrib))
    
    return np.concatenate(contributions_list, axis=0)

def visualize_feature_importance(model, data_loader, device, feature_names=None, top_k=20, save_dir="./visualizations"):
    os.makedirs(save_dir, exist_ok=True)

    contributions = get_feature_contributions(model, data_loader, device)
    abs_contributions = np.abs(contributions)
    feature_importance = abs_contributions.sum(axis=(0,1))
    feature_importance /= feature_importance.sum()

    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(len(feature_importance))]

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False).head(top_k)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='mako')
    plt.title('Top Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"feature_importance_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

def visualize_feature_importance_heatmap(model, data_loader, device, feature_names=None, top_k=10, save_dir="./visualizations"):
    os.makedirs(save_dir, exist_ok=True)

    contributions = get_feature_contributions(model, data_loader, device)
    abs_contributions = np.abs(contributions)
    time_feature_importance = abs_contributions.sum(axis=0)

    top_indices = np.argsort(time_feature_importance.sum(axis=0))[-top_k:]
    heatmap_data = time_feature_importance[:, top_indices]

    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in top_indices]
    else:
        feature_names = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, cmap="YlGnBu", xticklabels=feature_names,
                yticklabels=[f"t{i}" for i in range(heatmap_data.shape[0])],
                annot=True, fmt=".3f")
    plt.title('Feature Importance Heatmap')
    plt.xlabel('Feature')
    plt.ylabel('Time Step')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"feature_importance_heatmap_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

def visualize_all_feature_importance(model, data_loader, device, feature_names=None, save_dir="./visualizations"):
    visualize_feature_importance(model, data_loader, device, feature_names, save_dir=save_dir)
    visualize_feature_importance_heatmap(model, data_loader, device, feature_names, save_dir=save_dir)
