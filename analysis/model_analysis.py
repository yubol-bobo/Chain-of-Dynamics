#!/usr/bin/env python3
"""
Unified Model Analysis Script

Performs temporal, feature, and cross-temporal-feature analysis for any model
(RETAIN, TFCAM, HCTA, EnhancedTFCAM, etc.) that exposes get_attention_weights().
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import inspect
import datetime
import networkx as nx
try:
    from pyvis.network import Network
    _has_pyvis = True
except ImportError:
    _has_pyvis = False

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.retain import RETAIN
from models.tfcam import TFCAM
from models.ctga import CTGA


MODEL_MAP = {
    'retain': RETAIN,
    'tfcam': TFCAM,
    'ctga': CTGA
}


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_best_hyperparams(model_type):
    import yaml
    summary_path = f"Outputs/saved_models/{model_type.lower()}_hypersearch_summary.yaml"
    if not os.path.exists(summary_path):
        print(f"[WARNING] Best hyperparameter summary not found: {summary_path}")
        return None
    with open(summary_path, 'r') as f:
        summary = yaml.safe_load(f)
    return summary.get('best_params', None)

def load_model(model_type, config, checkpoint_path, device):
    ModelClass = MODEL_MAP[model_type.lower()]
    # Dynamically filter config to only pass valid args
    model_args = dict(
        input_dim=config['model']['input_dim'],
        emb_dim=config['model']['emb_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model'].get('num_heads', 1),
        num_layers=config['model'].get('num_layers', 1),
        output_dim=config['model'].get('output_dim', 1),
        dropout=config['model'].get('dropout', 0.2),
        max_seq_len=config['model'].get('max_seq_len', 50)
    )
    # Only keep args that are in the model's __init__
    valid_keys = inspect.signature(ModelClass.__init__).parameters.keys()
    filtered_args = {k: v for k, v in model_args.items() if k in valid_keys}
    model = ModelClass(**filtered_args)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def prepare_data(config):
    data = pd.read_csv(config['data']['processed_path']).fillna(0)
    input_dim = config['model']['input_dim']
    num_period = config['data']['month_count'] // 3
    features = data.drop(columns=['TMA_Acct', 'ESRD'], errors='ignore').values
    labels = data['ESRD'].values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = features.reshape(len(features), num_period, input_dim)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    from torch.utils.data import TensorDataset
    all_data = TensorDataset(features_tensor, labels_tensor)
    all_data_loader = DataLoader(all_data, batch_size=32, shuffle=False)
    feature_names = [f"Feature {i+1}" for i in range(input_dim)]
    time_points = [f"t-{i}" for i in range(num_period-1, -1, -1)]
    return all_data_loader, feature_names, time_points

def plot_temporal_attention(attn, time_points, save_path):
    attn_1d = np.array(attn).squeeze()
    if time_points is None or len(time_points) != len(attn_1d):
        time_points = [f"t-{i}" for i in range(len(attn_1d))]
    attn_1d = attn_1d[::-1]
    time_points = time_points[::-1]
    palette = sns.color_palette("viridis", len(time_points))[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=time_points, y=attn_1d, palette=palette)
    plt.title('Temporal Attention')
    plt.xlabel('Time')
    plt.ylabel('Attention Weight')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_attention(attn, feature_names, save_path, model=None, inputs=None, time_points=None, output_dir=None):
    attn = np.array(attn)
    if model is not None and hasattr(model, 'compute_contributions') and inputs is not None:
        try:
            contrib = model.compute_contributions(0)  # shape: [seq_len, input_dim]
            attn_1d = np.mean(contrib, axis=0)
            contrib_matrix = contrib  # [seq_len, input_dim]
            print(f"[DEBUG] Using compute_contributions for feature importance, shape: {attn_1d.shape}")
        except Exception as e:
            print(f"[WARNING] compute_contributions failed: {e}. Falling back to attention averaging.")
            attn_1d = None
            contrib_matrix = None
    else:
        attn_1d = None
        contrib_matrix = None
    if attn_1d is None:
        if attn.ndim > 1:
            attn_1d = attn.mean(axis=tuple(range(attn.ndim - 1)))
        else:
            attn_1d = attn
    print(f"[DEBUG] feature_names length: {len(feature_names)}, attn_1d shape: {attn_1d.shape}")
    if len(attn_1d) != len(feature_names):
        print(f"[WARNING] Feature attention shape {attn_1d.shape} does not match number of features ({len(feature_names)}). Skipping feature plot.")
        return
    feature_names = [str(f) for f in feature_names]
    # Horizontal bar plot for top-N features (absolute value)
    top_n = min(20, len(feature_names))
    abs_attn_1d = np.abs(attn_1d)
    sorted_indices = np.argsort(abs_attn_1d)[::-1]
    top_features = [feature_names[i] for i in sorted_indices[:top_n]]
    top_importance = abs_attn_1d[sorted_indices[:top_n]]
    colors = sns.color_palette("viridis", len(top_features))
    plt.figure(figsize=(12, 8))
    plt.barh(top_features[::-1], top_importance[::-1], color=colors)
    plt.xlabel('Absolute Importance')
    plt.title('Top Features (Across All Time Points)')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "feature_importance_top.png"), bbox_inches='tight')
    else:
        plt.savefig(save_path)
    plt.close()
    # Heatmap of feature contributions over time (if available)
    if contrib_matrix is not None and time_points is not None:
        plt.figure(figsize=(14, 8))
        sns.heatmap(contrib_matrix, cmap='RdBu_r', center=0, annot=False, xticklabels=feature_names, yticklabels=time_points)
        plt.xlabel('Features')
        plt.ylabel('Time Points')
        plt.title('Feature Contributions Over Time')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "feature_time_heatmap.png"), bbox_inches='tight')
        plt.close()
    # Top feature-time combinations
    if contrib_matrix is not None and time_points is not None:
        contribution_pairs = []
        for t, time in enumerate(time_points):
            for f, feature in enumerate(feature_names):
                contribution_pairs.append((f"{time}: {feature}", contrib_matrix[t, f]))
        contribution_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_pairs = contribution_pairs[:top_n]
        labels, values = zip(*top_pairs)
        max_val = max(abs(v) for v in values) if values else 1e-6
        pos_cmap = plt.cm.Blues
        neg_cmap = plt.cm.Reds
        colors = []
        for v in values:
            alpha = abs(v) / max_val
            if v >= 0:
                color = pos_cmap(alpha)
            else:
                color = neg_cmap(alpha)
            colors.append(color)
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_pairs)), [abs(v) for v in values], color=colors)
        plt.yticks(range(len(top_pairs)), labels)
        plt.xlabel('Absolute Contribution')
        plt.title('Top Feature-Time Combinations')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "top_feature_time_combinations.png"), bbox_inches='tight')
        plt.close()

def plot_cross_temporal_feature(attn_matrix, time_points, feature_names, save_path):
    plt.figure(figsize=(14, 10))
    sns.heatmap(attn_matrix, xticklabels=feature_names, yticklabels=time_points, cmap='viridis')
    plt.title('Cross-Temporal-Feature Attention')
    plt.xlabel('Feature')
    plt.ylabel('Time')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_inter_feature_influence_v2(model, data_loader, device, feature_names=None, time_points=None):
    """
    Vectorized cross-temporal-feature influence calculation.
    Returns a DataFrame (block matrix) with labels Feature_t-X.
    """
    model.eval()
    contributions_list = []
    cross_attn_list = []
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        batch_size, seq_len, input_dim = inputs.size()
        with torch.no_grad():
            _ = model(inputs)
            # Compute contributions for each sample in batch
            for i in range(batch_size):
                if not hasattr(model, 'compute_contributions'):
                    raise RuntimeError('Model does not support compute_contributions')
                C = model.compute_contributions(i)  # [seq_len, input_dim]
                contributions_list.append(C)
            # Get cross-attn weights (list of tensors)
            cross_attn = model.get_attention_weights().get('cross_feature', None)
            if isinstance(cross_attn, list):
                # Each element: [batch, num_heads, seq_len, seq_len]
                # Average over layers and heads
                batch_cross = []
                for cw in cross_attn:
                    # cw: [batch, num_heads, seq_len, seq_len]
                    batch_cross.append(cw.mean(dim=1))  # mean over heads -> [batch, seq_len, seq_len]
                # Stack over layers, then mean over layers
                batch_cross_tensor = torch.stack(batch_cross, dim=0).mean(dim=0)  # [batch, seq_len, seq_len]
            elif cross_attn is not None:
                batch_cross_tensor = cross_attn  # fallback
            else:
                raise RuntimeError('No cross_feature attention available')
            for i in range(batch_size):
                A = batch_cross_tensor[i].cpu().numpy()  # [seq_len, seq_len]
                cross_attn_list.append(A)
    contributions_array = np.stack(contributions_list, axis=0)  # [N, T, F]
    cross_attn_array = np.stack(cross_attn_list, axis=0)        # [N, T, T]
    N, T, F = contributions_array.shape
    # Swap axes for cross_attn_array to use A[n, u, t]
    cross_attn_swapped = np.transpose(cross_attn_array, (0, 2, 1))  # [N, T, T]
    # Vectorized influence calculation
    influence_tensor = np.einsum('ntu, ntf, nug -> ntufg', cross_attn_swapped, contributions_array, contributions_array)
    avg_influence = np.mean(influence_tensor, axis=0)  # [T, T, F, F]
    # Build block matrix
    M = np.full((T * F, T * F), np.nan)
    for t in range(T):
        for u in range(t + 1, T):
            row_start = t * F
            row_end = (t + 1) * F
            col_start = u * F
            col_end = (u + 1) * F
            M[row_start:row_end, col_start:col_end] = avg_influence[t, u, :, :]
    # Build labels
    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(F)]
    if time_points is None:
        time_points = [f"t-{i}" for i in range(T)]
    labels = [f"{feature_names[f]}_{time_points[t]}" for t in range(T) for f in range(F)]
    df_influence = pd.DataFrame(M, index=labels, columns=labels)
    return df_influence

def load_feature_names():
    import yaml
    with open('config/feature_names.yaml', 'r') as f:
        return yaml.safe_load(f)['feature_names']

def build_network_from_influence(df, threshold=0.001):
    print("[DEBUG] Influence DataFrame shape:", df.shape)
    print("[DEBUG] Influence DataFrame head:\n", df.head())
    G = nx.DiGraph()
    df_long = df.stack().reset_index()
    df_long.columns = ["Source", "Target", "Influence"]
    df_long.dropna(subset=["Influence"], inplace=True)
    df_long = df_long[df_long["Influence"].abs() >= threshold]
    print(f"[DEBUG] Number of edges to add (|influence| >= {threshold}):", len(df_long))
    for _, row in df_long.iterrows():
        source = row["Source"]
        target = row["Target"]
        influence = row["Influence"]
        G.add_node(source)
        G.add_node(target)
        G.add_edge(source, target, weight=influence)
    print("[DEBUG] Network nodes:", len(G.nodes()))
    print("[DEBUG] Network edges:", len(G.edges()))
    if len(G.edges()) == 0:
        print("[WARNING] The network has no edges. The HTML will be empty.")
    return G

def plot_interactive_network(G, output_file="interactive_network.html"):
    if not _has_pyvis:
        print("[WARNING] pyvis is not installed. Skipping interactive network visualization.")
        return
    net = Network(height="750px", width="100%", notebook=False, directed=True, cdn_resources="in_line")
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)
    for node in G.nodes():
        net.add_node(node, label=node)
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        edge_width = abs(weight) * 3
        edge_color = "red" if weight < 0 else "blue"
        net.add_edge(
            u, v,
            value=edge_width,
            width=edge_width,
            title=f"Value: {weight}",
            color=edge_color,
            physics=True,
            smooth=True
        )
    net.show_buttons(filter_=['physics'])
    net.show(output_file, notebook=False)

def compute_ctga_cross_feature_cov_corr(model, data_loader, feature_names=None, time_points=None, output_dir=None, model_name='ctga'):
    """
    For CTGA: Compute and save cross-feature covariance and correlation matrices for each time step, averaged across layers, projected back to feature space.
    Output as CSVs with block matrix structure (features x time).
    """
    import numpy as np
    import pandas as pd
    cov_matrices = []  # [layer][time][input_dim, input_dim]
    corr_matrices = []
    input_dim = model.input_dim
    emb_dim = model.emb_dim
    num_time = None
    embedding_weight = model.embedding.weight.detach().cpu().numpy()  # [emb_dim, input_dim]
    for inputs, _ in data_loader:
        inputs = inputs.to(next(model.parameters()).device)
        with torch.no_grad():
            _ = model(inputs)
            cross_feature_list = model.get_attention_weights().get('cross_feature', None)
            if not isinstance(cross_feature_list, list):
                print("[WARNING] CTGA cross_feature is not a list. Skipping.")
                return
            if num_time is None:
                num_time = cross_feature_list[0].shape[1]
            # For each layer
            for layer_idx, layer_hidden in enumerate(cross_feature_list):
                # layer_hidden: [batch, seq_len, emb_dim]
                time_covs = []
                time_corrs = []
                for t in range(layer_hidden.shape[1]):
                    features_t = layer_hidden[:, t, :].cpu().numpy()  # [batch, emb_dim]
                    # Project to feature space: [batch, emb_dim] @ [emb_dim, input_dim] = [batch, input_dim]
                    features_t_proj = features_t @ embedding_weight  # [batch, input_dim]
                    if features_t_proj.shape[0] < 2:
                        cov = np.full((input_dim, input_dim), np.nan)
                        corr = np.full((input_dim, input_dim), np.nan)
                    else:
                        cov = np.cov(features_t_proj, rowvar=False)
                        corr = np.corrcoef(features_t_proj, rowvar=False)
                    time_covs.append(cov)
                    time_corrs.append(corr)
                cov_matrices.append(time_covs)
                corr_matrices.append(time_corrs)
        break  # Only need one batch for analysis
    # Average across layers
    cov_matrices = np.array(cov_matrices)  # [num_layers, num_time, input_dim, input_dim]
    corr_matrices = np.array(corr_matrices)
    avg_cov = np.nanmean(cov_matrices, axis=0)  # [num_time, input_dim, input_dim]
    avg_corr = np.nanmean(corr_matrices, axis=0)
    # Build block matrix for CSV
    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(input_dim)]
    if time_points is None:
        time_points = [f"t-{i}" for i in range(num_time)]
    labels = [f"{feature_names[f]}_{time_points[t]}" for t in range(num_time) for f in range(input_dim)]
    block_cov = np.full((num_time * input_dim, num_time * input_dim), np.nan)
    block_corr = np.full((num_time * input_dim, num_time * input_dim), np.nan)
    for t in range(num_time):
        for u in range(num_time):
            row_start = t * input_dim
            row_end = (t + 1) * input_dim
            col_start = u * input_dim
            col_end = (u + 1) * input_dim
            if t >= u:  # Only fill lower triangle (causal half)
                block_cov[row_start:row_end, col_start:col_end] = avg_cov[t]
                block_corr[row_start:row_end, col_start:col_end] = avg_corr[t]
            # else: leave as NaN
    df_cov = pd.DataFrame(block_cov, index=labels, columns=labels)
    df_corr = pd.DataFrame(block_corr, index=labels, columns=labels)
    if output_dir is not None:
        cov_path = os.path.join(output_dir, f"{model_name}_cross_feature_covariance.csv")
        corr_path = os.path.join(output_dir, f"{model_name}_cross_feature_correlation.csv")
        df_cov.to_csv(cov_path)
        df_corr.to_csv(corr_path)
        print(f"[INFO] CTGA cross-feature covariance CSV saved to: {cov_path}")
        print(f"[INFO] CTGA cross-feature correlation CSV saved to: {corr_path}")
    return df_cov, df_corr

def main():
    parser = argparse.ArgumentParser(description='Unified Model Analysis Script')
    parser.add_argument('--model', type=str, required=True, help='Model type: retain, tfcam, hcta, enhanced_tfcam, mstca, ctga')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to best model checkpoint (if not provided, will use Outputs/saved_models/{model}_best_model_hypersearch.pt)')
    parser.add_argument('--config', type=str, required=False, help='Path to config YAML (if not provided, will use config/{model}_config.yaml)')
    parser.add_argument('--output', type=str, default='./visualizations/unified_analysis', help='Output directory')
    args = parser.parse_args()

    # Auto-determine checkpoint if not provided
    if args.checkpoint is None:
        args.checkpoint = f"Outputs/saved_models/{args.model.lower()}_best_model_hypersearch.pt"
        print(f"[INFO] Using checkpoint: {args.checkpoint}")

    # Auto-determine config if not provided
    if args.config is None:
        args.config = f"config/{args.model.lower()}_config.yaml"
        print(f"[INFO] Using config: {args.config}")

    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)

    

    # Load and apply best hyperparameters
    best_params = load_best_hyperparams(args.model)
    if best_params is not None:
        config['model'].update(best_params)
        print(f"[INFO] Loaded best hyperparameters from summary for {args.model}: {best_params}")

    device = torch.device(config['model'].get('device', 'cpu'))
    model = load_model(args.model, config, args.checkpoint, device)
    data_loader, feature_names, time_points = prepare_data(config)

    # Load feature names from config
    feature_names = load_feature_names()

    # Get a batch for analysis
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    with torch.no_grad():
        _ = model(inputs)
        if hasattr(model, 'get_attention_weights'):
            attn = model.get_attention_weights()
            print("[DEBUG] Attention keys and shapes:", {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in attn.items()})
        else:
            print(f"Model {args.model} does not implement get_attention_weights(). Skipping analysis.")
            return

    # Temporal-level analysis
    if 'temporal' in attn and attn['temporal'] is not None:
        temporal_attn = attn['temporal'].mean(dim=0).cpu().numpy() if hasattr(attn['temporal'], 'mean') else np.mean(attn['temporal'], axis=0)
        print(f"[DEBUG] Plotting temporal attention, shape: {temporal_attn.shape}")
        plot_temporal_attention(temporal_attn, time_points, os.path.join(args.output, f'{args.model}_temporal_attention.png'))
        print(f"Saved temporal attention plot: {args.model}_temporal_attention.png")
    else:
        print("No temporal attention available for this model.")

    # Feature-level analysis
    if 'feature' in attn and attn['feature'] is not None:
        feature_attn = attn['feature'].mean(dim=0).cpu().numpy() if hasattr(attn['feature'], 'mean') else np.mean(attn['feature'], axis=0)
        print(f"[DEBUG] Plotting feature attention, shape: {feature_attn.shape}")
        plot_feature_attention(feature_attn, feature_names, os.path.join(args.output, f'{args.model}_feature_attention.png'), model=model, inputs=inputs, time_points=time_points, output_dir=args.output)
        print(f"Saved feature attention plot: {args.model}_feature_attention.png")
    else:
        print("No feature attention available for this model.")

    # Cross-temporal-feature analysis (CSV only, vectorized, for all models with compute_contributions)
    df_influence = None
    if hasattr(model, 'compute_contributions'):
        try:
            print("[INFO] Running vectorized cross-temporal-feature influence analysis (CSV only)...")
            df_influence = compute_inter_feature_influence_v2(model, data_loader, device, feature_names, time_points)
            csv_path = os.path.join(args.output, f'{args.model}_cross_temporal_feature_influence.csv')
            df_influence.to_csv(csv_path, index=True)
            print(f"[INFO] Cross-temporal-feature influence CSV saved to: {csv_path}")
        except Exception as e:
            print(f"[WARNING] Cross-temporal-feature CSV analysis failed: {e}")
            df_influence = None
    else:
        print("No cross-temporal-feature analysis available for this model.")

    # CTGA-specific cross-feature analysis
    if args.model.lower() == 'ctga':
        print("[INFO] Running CTGA cross-feature covariance and correlation analysis...")
        compute_ctga_cross_feature_cov_corr(model, data_loader, feature_names, time_points, args.output, model_name=args.model.lower())

    # Save interactive network
    if df_influence is not None and _has_pyvis:
        print("[INFO] Generating interactive network visualization...")
        G = build_network_from_influence(df_influence, threshold=0.001)
        html_path = os.path.join(args.output, f'{args.model}_cross_temporal_feature_network.html')
        plot_interactive_network(G, output_file=html_path)
        print(f"[INFO] Interactive network saved to: {html_path}")
    elif df_influence is None:
        print("[INFO] No cross_feature attention available; skipping interactive network visualization.")
    else:
        print("[WARNING] pyvis not installed. Skipping interactive network visualization.")

if __name__ == "__main__":
    main() 