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

from analysis.analysis_utils import set_publication_style, load_config, load_trained_model, get_timestamp
from utils.train import prepare_data


def get_all_temporal(model, data_loader, device):
    """
    Collects temporal attention weights (stored in model.alpha_weights) for all samples.
    
    Parameters:
        model (RETAIN): The RETAIN model with 'alpha_weights' attribute.
        data_loader (DataLoader): DataLoader to iterate over the dataset.
        device (torch.device): Device to perform computations on.
        
    Returns:
        np.ndarray: A concatenated numpy array of shape [num_samples, seq_len] containing temporal weights.
    """
    model.eval()
    all_temporal_list = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            # Forward pass (populates model.alpha_weights)
            _ = model(inputs)
            
            # model.alpha_weights has shape [batch_size, seq_len, 1]
            temporal_batch = model.alpha_weights.squeeze(-1).cpu().numpy()  # shape: [batch_size, seq_len]
            all_temporal_list.append(temporal_batch)
    
    if not all_temporal_list:
        print("No temporal weights collected. Check the model and data loader.")
        return np.array([])
        
    all_temporal = np.concatenate(all_temporal_list, axis=0)
    return all_temporal



def plot_global_temporal_attention(model, data_loader, device, time_points=None, plot_type='box', 
                                   save_dir="./visualizations", reverse_time=False, timestamp=None):
    """
    Visualizes the temporal attention weights (alpha weights) across all samples for RETAIN model.
    Color is assigned based on importance values rather than time order.
    
    Parameters:
        model (RETAIN): The trained RETAIN model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        time_points (list, optional): Custom time step labels. If None, default labels are generated.
        plot_type (str): Type of plot to generate ('box', 'violin', 'bar', or 'line').
        save_dir (str): Directory to save the resulting plot.
        reverse_time (bool): Whether to reverse the time order in visualizations (t-n to t-0).
    """
    # Apply academic publication style settings
    set_publication_style(fontsize=20, label_fontweight='bold')
    
    # Gather temporal attention weights for all samples
    all_temporal = get_all_temporal(model, data_loader, device)
    
    if all_temporal.size == 0:
        print("Error: No temporal attention weights collected. Check the model and data loader.")
        return
        
    num_samples, seq_len = all_temporal.shape
    
    # Generate default time points if none are provided
    if time_points is None:
        if reverse_time:
            # Reversed order (t-n to t-0) - most recent visit is t-0
            time_points = [f"t-{i}" for i in range(seq_len - 1, -1, -1)]
        else:
            # Standard order (t0 to t7) - first visit is t0
            time_points = [f"t{i}" for i in range(seq_len)]
    
    # If we're reversing the time order, we need to reverse the columns
    if reverse_time:
        all_temporal = all_temporal[:, ::-1]
    
    # Create a DataFrame for easier plotting
    df_temporal = pd.DataFrame(all_temporal, columns=time_points)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot based on the selected type
    if plot_type == 'box':
        df_melt = df_temporal.melt(var_name='Time Step', value_name='Attention Weight')
        ax = sns.boxplot(x='Time Step', y='Attention Weight', data=df_melt)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attention Weight')
        plt.xticks(rotation=45)
    
    elif plot_type == 'violin':
        df_melt = df_temporal.melt(var_name='Time Step', value_name='Attention Weight')
        ax = sns.violinplot(x='Time Step', y='Attention Weight', data=df_melt)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attention Weight')
        plt.xticks(rotation=45)
    
    elif plot_type == 'bar':
        # Plot the mean temporal attention across samples
        mean_temporal = df_temporal.mean(axis=0)
        
        # Sort by importance value for color assignment
        sorted_indices = np.argsort(mean_temporal.values)
        sorted_time_points = [time_points[i] for i in sorted_indices]
        sorted_values = mean_temporal.values[sorted_indices]
        
        # Create color palette based on sorted importance values
        # Intentionally using non-inversed ordering - lower importance gets darker colors
        palette = sns.color_palette("viridis", len(time_points))
        
        # Create a mapping from time point to color based on importance
        # Lower importance values get darker colors (rank corresponds directly to color index)
        color_mapping = {time_points[i]: palette[rank] for rank, i in enumerate(sorted_indices)}
        
        # Plot with original time order but colors based on importance
        fig, ax = plt.subplots(figsize=(14, 10))
        bars = ax.bar(time_points, mean_temporal.values, color=[color_mapping[t] for t in time_points])
        
        plt.xlabel('Time Step')
        plt.ylabel('Mean Attention Weight')
        
        # Bold tick labels for a cleaner look
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
        
        # Add a colorbar legend to explain the color mapping
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(min(mean_temporal.values), max(mean_temporal.values)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        # cbar.set_label('Attention Weight Magnitude (Darker = Lower)')
    
    elif plot_type == 'line':
        # Plot the mean temporal attention as a line plot
        mean_temporal = df_temporal.mean(axis=0)
        
        # Sort by importance to assign colors
        sorted_indices = np.argsort(mean_temporal.values)
        color_values = np.linspace(0, 1, len(time_points))
        color_mapping = {time_points[i]: plt.cm.viridis(color_values[rank]) 
                         for rank, i in enumerate(sorted_indices)}
        
        # Create figure and axis objects explicitly
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot each point with color based on its importance
        for i, t in enumerate(time_points):
            if i > 0:
                ax.plot([time_points[i-1], t], 
                         [mean_temporal[time_points[i-1]], mean_temporal[t]], 
                         color='gray', linewidth=1.5)
            ax.scatter(t, mean_temporal[t], color=color_mapping[t], s=150, zorder=5)
            
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean Attention Weight')
        plt.xticks(rotation=45)
        
        # Add a colorbar legend
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(min(mean_temporal.values), max(mean_temporal.values)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Weight Magnitude')
    
    # Remove any title for a cleaner academic style
    plt.title('')
    
    # Ensure the save directory exists and then save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"retain_temporal_attention_{plot_type}_{timestamp}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    
    print(f"Plot saved to: {save_path}")
    plt.show()
    
    # Also save the data for future reference
    data_path = os.path.join(save_dir, f"retain_temporal_attention_data_{timestamp}.csv")
    df_temporal.to_csv(data_path, index=False)
    print(f"Data saved to: {data_path}")


def plot_temporal_comparison(model, data_loader, device, time_points=None, 
                                    save_dir="./visualizations", reverse_time=False, timestamp=None):
    """
    Compares temporal attention weights between positive and negative outcome groups.
    Color assignment is based on importance values rather than time order.
    
    Parameters:
        model (RETAIN): The trained RETAIN model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        time_points (list, optional): Custom time step labels. If None, default labels are generated.
        save_dir (str): Directory to save the resulting plot.
        reverse_time (bool): Whether to reverse the time order in visualizations.
    """
    # Apply academic publication style settings
    set_publication_style(fontsize=20, label_fontweight='bold')
    
    # Collect all temporal weights and corresponding labels
    model.eval()
    all_temporal_list = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            # Forward pass to populate alpha_weights
            _ = model(inputs)
            
            # Extract alpha weights
            temporal_batch = model.alpha_weights.squeeze(-1).cpu().numpy()
            all_temporal_list.append(temporal_batch)
            all_labels.append(labels.numpy())
    
    if not all_temporal_list:
        print("Error: No temporal attention weights collected. Check the model and data loader.")
        return
        
    # Concatenate all batches
    all_temporal = np.concatenate(all_temporal_list, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    num_samples, seq_len = all_temporal.shape
    
    # Generate default time points if none are provided
    if time_points is None:
        if reverse_time:
            # Reversed order (t-n to t-0)
            time_points = [f"t-{i}" for i in range(seq_len - 1, -1, -1)]
        else:
            # Standard order (t0 to t7)
            time_points = [f"t{i}" for i in range(seq_len)]
    
    # If we're reversing the time order, we need to reverse the columns
    if reverse_time:
        all_temporal = all_temporal[:, ::-1]
    
    # Separate positive and negative samples
    pos_temporal = all_temporal[all_labels == 1]
    neg_temporal = all_temporal[all_labels == 0]
    
    # Calculate means for both groups
    pos_mean = pos_temporal.mean(axis=0)
    neg_mean = neg_temporal.mean(axis=0)
    
    # Create a figure for line plot comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color mapping based on importance for positive group
    pos_sorted_indices = np.argsort(pos_mean)
    pos_color_values = np.linspace(0, 1, len(time_points))
    pos_color_mapping = {time_points[i]: plt.cm.Reds(pos_color_values[len(time_points)-1-rank]) 
                     for rank, i in enumerate(pos_sorted_indices)}
    
    # Color mapping based on importance for negative group
    neg_sorted_indices = np.argsort(neg_mean)
    neg_color_values = np.linspace(0, 1, len(time_points))
    neg_color_mapping = {time_points[i]: plt.cm.Blues(neg_color_values[len(time_points)-1-rank]) 
                     for rank, i in enumerate(neg_sorted_indices)}
    
    # Plot the lines connecting points
    ax.plot(time_points, pos_mean, 'r-', linewidth=1.5, alpha=0.7, label='Positive Outcome')
    ax.plot(time_points, neg_mean, 'b-', linewidth=1.5, alpha=0.7, label='Negative Outcome')
    
    # Plot individual points with colors based on importance
    for i, t in enumerate(time_points):
        ax.scatter(t, pos_mean[i], color=pos_color_mapping[t], s=120, zorder=5, edgecolor='black', linewidth=0.5)
        ax.scatter(t, neg_mean[i], color=neg_color_mapping[t], s=120, zorder=5, edgecolor='black', linewidth=0.5, marker='s')
    
    # Add shaded areas for standard error
    pos_stderr = pos_temporal.std(axis=0) / np.sqrt(pos_temporal.shape[0])
    neg_stderr = neg_temporal.std(axis=0) / np.sqrt(neg_temporal.shape[0])
    
    ax.fill_between(time_points, pos_mean - pos_stderr, pos_mean + pos_stderr, color='r', alpha=0.2)
    ax.fill_between(time_points, neg_mean - neg_stderr, neg_mean + neg_stderr, color='b', alpha=0.2)
    
    # Set labels and legend
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean Attention Weight')
    ax.legend(loc='best', frameon=True, framealpha=0.8, fontsize=18)
    
    # Add text annotations indicating color meaning
    ax.text(0.02, 0.98, " ", 
             transform=ax.transAxes, fontsize=16, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Grid and spines
    ax.grid(True, alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Ensure the save directory exists and then save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"retain_temporal_comparison_{timestamp}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    
    print(f"Comparison plot saved to: {save_path}")
    plt.show()
    
    # Create a more detailed visualization with error bars and color coded by importance
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # For bar chart, prepare data
    pos_ranks = np.argsort(np.argsort(pos_mean))  # Ranking for positive
    neg_ranks = np.argsort(np.argsort(neg_mean))  # Ranking for negative
    
    # Map ranks to colors
    pos_colors = [plt.cm.Reds(0.3 + 0.7 * (rank / (len(time_points) - 1))) for rank in pos_ranks]
    neg_colors = [plt.cm.Blues(0.3 + 0.7 * (rank / (len(time_points) - 1))) for rank in neg_ranks]
    
    # Set up x positions
    x = np.arange(len(time_points))
    width = 0.35
    
    # Create bars with colors based on importance
    ax.bar(x - width/2, pos_mean, width, color=pos_colors, label='Positive', 
            edgecolor='black', linewidth=0.5, alpha=0.9)
    ax.bar(x + width/2, neg_mean, width, color=neg_colors, label='Negative',
            edgecolor='black', linewidth=0.5, alpha=0.9)
    
    # Add error bars
    ax.errorbar(x - width/2, pos_mean, yerr=pos_stderr, fmt='none', ecolor='black', capsize=5, alpha=0.7)
    ax.errorbar(x + width/2, neg_mean, yerr=neg_stderr, fmt='none', ecolor='black', capsize=5, alpha=0.7)
    
    # Set labels, legend, and ticks
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean Attention Weight')
    ax.legend(title='Outcome', loc='best', frameon=True, framealpha=0.8, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(time_points)
    
    # Add explanation text
    ax.text(0.02, 0.98, "", 
             transform=ax.transAxes, fontsize=16, va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the bar plot
    bar_save_path = os.path.join(save_dir, f"retain_temporal_comparison_bar_{timestamp}.png")
    plt.savefig(bar_save_path, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()
    
    print(f"Bar comparison plot saved to: {bar_save_path}")
    
    # Also save the data for future reference
    data_dict = {
        'time_step': time_points,
        'positive_mean': pos_mean,
        'positive_stderr': pos_stderr,
        'negative_mean': neg_mean,
        'negative_stderr': neg_stderr,
        'positive_importance_rank': pos_ranks,
        'negative_importance_rank': neg_ranks
    }
    df_data = pd.DataFrame(data_dict)
    data_path = os.path.join(save_dir, f"retain_temporal_comparison_data_{timestamp}.csv")
    df_data.to_csv(data_path, index=False)
    
    print(f"Comparison data saved to: {data_path}")

def main(timestamp=None):
    if timestamp is None:
        timestamp = get_timestamp()

    config = load_config('config/config.yaml')
    hyperparam_path = 'Outputs/best_hyperparameters.yaml'
    model_path = 'Outputs/retain_best_model.pt'
    
    device = torch.device(config['model']['device'])
    model = load_trained_model(config, model_path, hyperparam_path)

    train_data, val_data, test_data = prepare_data(config)
    combined_dataset = torch.utils.data.ConcatDataset([train_data, val_data, test_data])
    all_data_loader = DataLoader(combined_dataset, batch_size=config['model']['batch_size'], shuffle=False)


    #plot_global_temporal_attention(model, test_loader, device)
    
    # For visualizing temporal attention across all samples
    plot_global_temporal_attention(
        model=model, 
        data_loader=all_data_loader, 
        device=device, 
        plot_type='bar',  # Options: 'box', 'violin', 'bar', 'line'
        save_dir='./visualizations',
        timestamp=timestamp  # Use the timestamp for file naming
    )

    # For comparing temporal attention between positive and negative outcomes
    plot_temporal_comparison(
        model=model, 
        data_loader=all_data_loader, 
        device=device, 
        save_dir='./visualizations',
        timestamp=timestamp 
    )


if __name__ == '__main__':
    main()
