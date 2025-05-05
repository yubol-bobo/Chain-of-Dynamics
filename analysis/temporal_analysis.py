import os
import torch
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime

# Custom module imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.analysis_utils import set_publication_style, load_config, load_trained_model
from utils.train import prepare_data


def get_all_temporal(model, data_loader, device):
    model.eval()
    all_temporal = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            all_temporal.append(model.alpha_weights.squeeze(-1).cpu().numpy())
    return np.concatenate(all_temporal, axis=0)


def plot_global_temporal_attention(model, data_loader, device, save_dir="./visualizations"):
    set_publication_style()

    temporal_weights = get_all_temporal(model, data_loader, device)
    time_steps = [f"t-{i}" for i in reversed(range(temporal_weights.shape[1]))]
    df_temporal = pd.DataFrame(temporal_weights, columns=time_steps)

    plt.figure()
    sns.boxplot(data=df_temporal)
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{save_dir}/temporal_attention_{timestamp}.png", dpi=300)
    plt.close()

    df_temporal.to_csv(f"{save_dir}/temporal_attention_{timestamp}.csv", index=False)


def main():

    config = load_config('config/config.yaml')
    hyperparam_path = 'Outputs/best_hyperparameters.yaml'
    model_path = 'Outputs/retain_best_model.pt'
    
    device = torch.device(config['model']['device'])
    model = load_trained_model(config, model_path, hyperparam_path)

    _, _, test_data = prepare_data(config)
    test_loader = DataLoader(test_data, batch_size=config['model']['batch_size'])

    plot_global_temporal_attention(model, test_loader, device)


if __name__ == '__main__':
    main()
