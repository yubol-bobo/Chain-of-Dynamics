#!/usr/bin/env python3
"""
Unified Model Analysis Script

Performs temporal, feature, and cross-temporal-feature analysis for any model
(RETAIN, CoI, BiLSTM) that exposes get_attention_weights().
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

from src.models.retain import RETAIN
from src.models.coi import CoI
from src.models.bilstm import BiLSTM


MODEL_MAP = {
    'retain': RETAIN,
    'coi': CoI,
    'bilstm': BiLSTM
}


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_best_hyperparams(model_type, results_path=None):
    import yaml
    if results_path is None:
        results_path = os.path.join("results", "ckd")
    summary_path = os.path.join(results_path, f"{model_type.lower()}_hypersearch_summary.yaml")
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
    import os
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    model_type = config['model'].get('type', 'model')
    results_path = config['paths']['results_path']
    X_test_path = os.path.join(results_path, f"{model_type}_X_test_imputed.npy")
    y_test_path = os.path.join(results_path, f"{model_type}_y_test_imputed.npy")
    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        raise FileNotFoundError(f"Imputed/scaled test set not found: {X_test_path} or {y_test_path}. Please run train.py first.")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    def to_tensor(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return torch.as_tensor(x, dtype=torch.float32)
    X_test = to_tensor(X_test)
    y_test = to_tensor(y_test)
    test_data = TensorDataset(X_test, y_test)
    test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    seq_len = X_test.shape[1]
    input_dim = X_test.shape[2]
    feature_names = [f"Feature {i+1}" for i in range(input_dim)]
    time_points = [f"t-{i}" for i in range(seq_len - 1, -1, -1)]
    return X_test, y_test, test_data_loader, feature_names, time_points, input_dim

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

def plot_feature_attention(attn, feature_names, save_path, model=None, inputs=None, time_points=None, output_dir=None, contrib_matrix_override=None):
    attn = np.array(attn)
    if contrib_matrix_override is not None:
        contrib_matrix = np.array(contrib_matrix_override)
        attn_1d = np.mean(contrib_matrix, axis=0)
    elif model is not None and hasattr(model, 'compute_contributions') and inputs is not None:
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

def evaluate_model(model, data_loader, device):
    all_preds = []
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
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
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)

def select_patient_indices(y_test, n_patients=8, seed=2025):
    rng = np.random.default_rng(seed)
    y_np = y_test.cpu().numpy().astype(int)
    pos_idx = np.where(y_np == 1)[0]
    neg_idx = np.where(y_np == 0)[0]
    half = n_patients // 2
    n_pos = min(len(pos_idx), half)
    n_neg = min(len(neg_idx), n_patients - n_pos)
    pos_sel = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    neg_sel = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
    selected = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(selected)
    return selected.tolist()

def compute_inter_feature_influence_single(model, inputs, feature_names=None, time_points=None):
    """
    Cross-temporal-feature influence for a single patient.
    Returns a block-matrix DataFrame with Feature_t-X labels.
    """
    model.eval()
    with torch.no_grad():
        _ = model(inputs)
        contrib = model.compute_contributions(0)  # [T, F]
        cross_attn = model.get_attention_weights().get('cross_feature', None)
        if isinstance(cross_attn, list) and len(cross_attn) > 0:
            # Average over heads, then layers -> [1, T, T] -> [T, T]
            per_layer = [cw.mean(dim=1) for cw in cross_attn]
            cross_attn_tensor = torch.stack(per_layer, dim=0).mean(dim=0)[0]
        elif cross_attn is not None:
            if cross_attn.dim() == 4:
                cross_attn_tensor = cross_attn.mean(dim=1)[0]
            else:
                cross_attn_tensor = cross_attn[0]
        else:
            raise RuntimeError('No cross_feature attention available')
    A = cross_attn_tensor.detach().cpu().numpy()  # [T, T]
    A_swapped = A.T  # match compute_inter_feature_influence_v2 convention
    T, F = contrib.shape
    influence = np.full((T * F, T * F), np.nan, dtype=float)
    for t in range(T):
        for u in range(t + 1, T):
            block = A_swapped[t, u] * np.outer(contrib[t], contrib[u])
            rs, re = t * F, (t + 1) * F
            cs, ce = u * F, (u + 1) * F
            influence[rs:re, cs:ce] = block
    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(F)]
    if time_points is None:
        time_points = [f"t-{i}" for i in range(T)]
    labels = [f"{feature_names[f]}_{time_points[t]}" for t in range(T) for f in range(F)]
    return pd.DataFrame(influence, index=labels, columns=labels)

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



def main():
    parser = argparse.ArgumentParser(description='Unified Model Analysis Script')
    parser.add_argument('--model', type=str, required=True, help='Model type: retain, coi, bilstm, transformer, adacare, stagenet')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to best model checkpoint (if not provided, will use results_path/{model}_best_model_hypersearch.pt from the config)')
    parser.add_argument('--config', type=str, required=False, help='Path to config YAML (if not provided, will use config/{model}_config.yaml)')
    parser.add_argument('--output', type=str, default='./visualizations/unified_analysis', help='Output directory')
    args = parser.parse_args()

    # Auto-determine config if not provided
    if args.config is None:
        args.config = f"config/{args.model.lower()}_config.yaml"
        print(f"[INFO] Using config: {args.config}")

    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)
    config['model']['type'] = args.model.lower()
    results_path = config['paths']['results_path']

    # Auto-determine checkpoint if not provided (use results path).
    if args.checkpoint is None:
        args.checkpoint = os.path.join(results_path, f"{args.model.lower()}_best_model_hypersearch.pt")
        print(f"[INFO] Using checkpoint: {args.checkpoint}")

    

    # Load and apply best hyperparameters
    best_params = load_best_hyperparams(args.model, results_path=results_path)
    if best_params is not None:
        config['model'].update(best_params)
        print(f"[INFO] Loaded best hyperparameters from summary for {args.model}: {best_params}")

    X_test, y_test, data_loader, _, time_points, input_dim = prepare_data(config)
    config['model']['input_dim'] = input_dim
    device = torch.device(config['model'].get('device', 'cpu'))
    model = load_model(args.model, config, args.checkpoint, device)

    # Load feature names from config
    feature_names = load_feature_names()
    if len(feature_names) != input_dim:
        print(f"[WARNING] feature_names length ({len(feature_names)}) does not match input_dim ({input_dim}). Using generic names.")
        feature_names = [f"Feature {i+1}" for i in range(input_dim)]

    # 1) Cohort-level analysis
    cohort_dir = os.path.join(args.output, "cohort")
    os.makedirs(cohort_dir, exist_ok=True)

    all_labels, all_preds, all_probs = evaluate_model(model, data_loader, device)
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score
    f1 = f1_score(all_labels, all_preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auroc = float('nan')
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    print(f"[COHORT] F1: {f1:.4f}, AUROC: {auroc:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}")

    import yaml
    cohort_metrics = {
        'f1': float(f1),
        'auroc': float(auroc),
        'accuracy': float(acc),
        'precision': float(prec)
    }
    cohort_metrics_path = os.path.join(cohort_dir, 'metrics_test.yaml')
    with open(cohort_metrics_path, 'w') as f:
        yaml.dump(cohort_metrics, f)

    temporal_batches = []
    feature_batches = []
    contrib_list = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            if not hasattr(model, 'get_attention_weights'):
                continue
            attn = model.get_attention_weights()
            temporal = attn.get('temporal', None)
            if temporal is not None:
                temporal_batches.append(temporal.mean(dim=0).detach().cpu().numpy())
            feature = attn.get('feature', None)
            if feature is not None:
                feature_batches.append(feature.mean(dim=0).detach().cpu().numpy())
            if hasattr(model, 'compute_contributions'):
                for i in range(inputs.size(0)):
                    contrib_list.append(model.compute_contributions(i))

    if temporal_batches:
        temporal_cohort = np.mean(np.stack(temporal_batches, axis=0), axis=0)
        plot_temporal_attention(temporal_cohort, time_points, os.path.join(cohort_dir, f'{args.model}_temporal_attention.png'))
    else:
        print("[COHORT] No temporal attention available.")

    if contrib_list:
        contrib_cohort = np.mean(np.stack(contrib_list, axis=0), axis=0)  # [T, F]
        plot_feature_attention(
            contrib_cohort, feature_names, os.path.join(cohort_dir, f'{args.model}_feature_attention.png'),
            time_points=time_points, output_dir=cohort_dir, contrib_matrix_override=contrib_cohort
        )
    elif feature_batches:
        feature_cohort = np.mean(np.stack(feature_batches, axis=0), axis=0)
        plot_feature_attention(
            feature_cohort, feature_names, os.path.join(cohort_dir, f'{args.model}_feature_attention.png'),
            time_points=time_points, output_dir=cohort_dir
        )
    else:
        print("[COHORT] No feature attention available.")

    df_influence = None
    if hasattr(model, 'compute_contributions'):
        try:
            df_influence = compute_inter_feature_influence_v2(model, data_loader, device, feature_names, time_points)
            cohort_csv_path = os.path.join(cohort_dir, f'{args.model}_cross_temporal_feature_influence.csv')
            df_influence.to_csv(cohort_csv_path, index=True)
        except Exception as e:
            print(f"[COHORT] Cross-temporal-feature analysis failed: {e}")
            df_influence = None

    if df_influence is not None and _has_pyvis:
        G = build_network_from_influence(df_influence, threshold=0.001)
        cohort_html_path = os.path.join(cohort_dir, f'{args.model}_cross_temporal_feature_network.html')
        plot_interactive_network(G, output_file=cohort_html_path)

    # 2) Individual-level analysis for 8 patients (mix of positive/negative labels)
    patient_indices = select_patient_indices(y_test, n_patients=8, seed=2025)
    print(f"[PATIENT] Selected indices: {patient_indices}")

    for idx in patient_indices:
        label = int(y_test[idx].item())
        patient_dir = os.path.join(args.output, f"patient_{idx}_label_{label}")
        os.makedirs(patient_dir, exist_ok=True)
        inputs = X_test[idx:idx + 1].to(device)

        with torch.no_grad():
            _ = model(inputs)
            if not hasattr(model, 'get_attention_weights'):
                continue
            attn = model.get_attention_weights()

        temporal = attn.get('temporal', None)
        if temporal is not None:
            temporal_attn = temporal[0].detach().cpu().numpy()
            plot_temporal_attention(temporal_attn, time_points, os.path.join(patient_dir, f'{args.model}_temporal_attention.png'))

        feature = attn.get('feature', None)
        if feature is not None:
            feature_attn = feature[0].detach().cpu().numpy()
            plot_feature_attention(
                feature_attn, feature_names, os.path.join(patient_dir, f'{args.model}_feature_attention.png'),
                model=model, inputs=inputs, time_points=time_points, output_dir=patient_dir
            )

        if hasattr(model, 'compute_contributions'):
            try:
                df_single = compute_inter_feature_influence_single(model, inputs, feature_names, time_points)
                single_csv_path = os.path.join(patient_dir, f'{args.model}_cross_temporal_feature_influence.csv')
                df_single.to_csv(single_csv_path, index=True)
                if _has_pyvis:
                    G_single = build_network_from_influence(df_single, threshold=0.001)
                    single_html_path = os.path.join(patient_dir, f'{args.model}_cross_temporal_feature_network.html')
                    plot_interactive_network(G_single, output_file=single_html_path)
            except Exception as e:
                print(f"[PATIENT {idx}] Cross-temporal-feature analysis failed: {e}")

if __name__ == "__main__":
    main() 
