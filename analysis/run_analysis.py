
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
from torch.utils.data import DataLoader
from models.retain import RETAIN
from analysis.temporal_analysis import plot_global_temporal_attention, plot_temporal_comparison
from analysis.feature_analysis import visualize_all_feature_importance
from utils.hyperparameter_tuning import prepare_data

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()

    # Device setup
    device = torch.device(config['model']['device'])

    # Load best model parameters
    best_model_path = os.path.join(config['paths']['save_path'], 'best_model.pt')
    model = RETAIN(
        input_dim=config['model']['input_dim'],
        emb_dim=config['model']['emb_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Prepare data
    train_data, val_data, test_data = prepare_data(config)
    all_data = torch.utils.data.ConcatDataset([train_data, val_data, test_data])
    data_loader = DataLoader(all_data, batch_size=64, shuffle=False)

    # Define your feature names list if you have one
    feature_names = [f"Feature {i+1}" for i in range(config['model']['input_dim'])]

    # Run Temporal Analysis
    plot_global_temporal_attention(model, data_loader, device)
    plot_temporal_comparison(model, data_loader, device)

    # Run Feature Analysis
    visualize_all_feature_importance(model, data_loader, device, feature_names)

if __name__ == '__main__':
    main()

