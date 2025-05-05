import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
from analysis_utils  import set_publication_style, load_config, load_trained_model
from train import prepare_data

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

feature_list = [
    'Diabetes', 'Htn', 'Cvd', 'Anemia', 'MA', 'Prot', 'SH', 'Phos', 'Athsc', 'CHF',
    'Stroke', 'CD', 'MI', 'FE', 'MD', 'ND', 'S4', 'S5', 'Serum_Calcium', 'eGFR', 
    'Phosphorus', 'Intact_PTH', 'Hemoglobin', 'UACR', 'Age', 'Gender', 'Race', 'BMI',
    'n_claims_DR', 'n_claims_I', 'n_claims_O', 'n_claims_P', 'net_exp_DR', 'net_exp_I', 
    'net_exp_O', 'net_exp_P'
]

def get_feature_weights(model, data_loader, device):
    model.eval()
    all_contributions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            model(inputs)
            batch_contrib = [model.compute_contributions(i) for i in range(inputs.size(0))]
            all_contributions.extend(batch_contrib)
    return np.array(all_contributions)

def compute_feature_importance(contributions):
    abs_contrib = np.abs(contributions)
    importance = abs_contrib.sum(axis=(0,1))
    return importance / importance.sum()

def visualize_feature_importance(importance, save_dir="./visualizations"):
    set_publication_style()
    df = pd.DataFrame({'Feature': feature_list, 'Importance': importance})
    df = df.sort_values('Importance', ascending=False).head(20)

    plt.figure()
    sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/feature_importance_{timestamp}.png", dpi=300)
    plt.close()

    df.to_csv(f"{save_dir}/feature_importance_{timestamp}.csv", index=False)

def main():
    config = load_config('config/config.yaml')
    device = torch.device(config['model']['device'])
    model = load_trained_model(config, './models/retain_best_model.pt')

    _, _, test_data = prepare_data(config)
    test_loader = DataLoader(test_data, batch_size=config['model']['batch_size'])

    contributions = get_feature_weights(model, test_loader, device)
    importance = compute_feature_importance(contributions)
    visualize_feature_importance(importance)

if __name__ == '__main__':
    main()
