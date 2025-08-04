# Chain-of-Influence Paper Visualization Guide

This guide helps you generate the visualizations referenced in the AAAI paper draft.

## Required Figures

### 1. Temporal Attention Comparison (`temporal_attention_comparison.png`)
- **Purpose**: Compare temporal attention patterns between RETAIN and CoI
- **Data needed**: Temporal attention weights from both models
- **Size**: 0.48\textwidth (approximately 3.5 inches wide)

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_temporal_attention_comparison(retain_weights, coi_weights, time_points):
    """
    Plot temporal attention comparison between RETAIN and CoI
    
    Args:
        retain_weights: Array of temporal attention weights from RETAIN
        coi_weights: Array of temporal attention weights from CoI  
        time_points: Array of time point labels (e.g., [3, 6, 9, 12, 15, 18, 21, 24])
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # RETAIN attention
    ax1.bar(time_points, retain_weights, alpha=0.7, color='skyblue')
    ax1.set_title('(a) RETAIN Temporal Attention', fontsize=12)
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Attention Weight')
    ax1.set_ylim(0, max(max(retain_weights), max(coi_weights)) * 1.1)
    
    # CoI attention
    ax2.bar(time_points, coi_weights, alpha=0.7, color='lightcoral')
    ax2.set_title('(b) CoI Temporal Attention', fontsize=12)
    ax2.set_xlabel('Time (months)')
    ax2.set_ylabel('Attention Weight')
    ax2.set_ylim(0, max(max(retain_weights), max(coi_weights)) * 1.1)
    
    plt.tight_layout()
    plt.savefig('temporal_attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# Extract from your model analysis results
# retain_attention = model_retain.temporal_weights  # Shape: [batch_size, seq_len] -> average over batch
# coi_attention = model_coi.temporal_weights      # Shape: [batch_size, seq_len] -> average over batch
# time_points = [3, 6, 9, 12, 15, 18, 21, 24]
# plot_temporal_attention_comparison(retain_attention, coi_attention, time_points)
```

### 2. Influence Chains Visualization (`influence_chains.png`)
- **Purpose**: Show the top influence chains discovered by CoI
- **Data needed**: Chain-of-influence tensor I[t, i, t', j]
- **Size**: 0.48\textwidth

```python
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def plot_influence_chains(influence_data, feature_names, time_points, top_k=10):
    """
    Plot the strongest influence chains as a directed graph
    
    Args:
        influence_data: Dictionary with influence chains
        feature_names: List of feature names
        time_points: List of time points
        top_k: Number of top influences to show
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges for top influences
    influences = []
    for (t1, feat1, t2, feat2), strength in influence_data.items():
        influences.append(((t1, feat1, t2, feat2), strength))
    
    # Sort by strength and take top_k
    influences.sort(key=lambda x: x[1], reverse=True)
    top_influences = influences[:top_k]
    
    # Add nodes and edges
    for (t1, feat1, t2, feat2), strength in top_influences:
        node1 = f"{feature_names[feat1]}\n(Month {time_points[t1]})"
        node2 = f"{feature_names[feat2]}\n(Month {time_points[t2]})"
        G.add_edge(node1, node2, weight=strength)
    
    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw edges with thickness proportional to influence strength
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    edge_widths = [5 * (w / max_weight) for w in edge_weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, arrowsize=20)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', 
                          alpha=0.8)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    ax.set_title('Top Influence Chains in CoI Model', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('influence_chains.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# influence_data = {
#     (1, 20, 4, 23): 0.45,  # eGFR at month 6 -> Hemoglobin at month 15
#     (0, 1, 2, 20): 0.38,   # Diabetes at month 3 -> eGFR at month 9
#     # ... more influences
# }
# plot_influence_chains(influence_data, feature_names, [3, 6, 9, 12, 15, 18, 21, 24])
```

## Generating Visualizations from Your Models

### Step 1: Extract Attention Weights

```python
# From your analysis/model_analysis.py, extract attention weights
def extract_attention_weights(model, data_loader, model_name):
    """Extract attention weights from trained model"""
    model.eval()
    temporal_weights_all = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Extract temporal attention weights
            if hasattr(model, 'temporal_weights'):
                temporal_weights_all.append(model.temporal_weights.cpu().numpy())
    
    # Average over all batches and samples
    temporal_weights = np.concatenate(temporal_weights_all, axis=0)
    avg_temporal_weights = np.mean(temporal_weights, axis=0)
    
    return avg_temporal_weights
```

### Step 2: Compute Chain-of-Influence

```python
def compute_chain_influence(model, data_loader, top_k=50):
    """Compute chain-of-influence from CoI model"""
    model.eval()
    
    all_influences = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get contribution matrix C and attention matrix A
            C = model.local_contributions  # [batch, time, features]
            A = model.cross_attention_matrix  # [batch, time, time]
            
            # Compute influence tensor I[t,i,t',j] = C[t,i] * A[t,t'] * C[t',j]
            batch_size, seq_len, n_features = C.shape
            
            for b in range(batch_size):
                for t in range(seq_len):
                    for t_prime in range(t+1, seq_len):
                        for i in range(n_features):
                            for j in range(n_features):
                                influence = C[b,t,i] * A[b,t,t_prime] * C[b,t_prime,j]
                                all_influences.append(((t, i, t_prime, j), influence.item()))
    
    # Sort by influence strength and return top_k
    all_influences.sort(key=lambda x: abs(x[1]), reverse=True)
    return dict(all_influences[:top_k])
```

### Step 3: Update Your analysis/model_analysis.py

Add this function to generate paper figures:

```python
def generate_paper_figures(model_retain, model_coi, test_data, feature_names, output_dir):
    """Generate all figures needed for the paper"""
    
    # 1. Extract temporal attention weights
    retain_temporal = extract_attention_weights(model_retain, test_data, 'retain')
    coi_temporal = extract_attention_weights(model_coi, test_data, 'coi')
    
    time_points = [3, 6, 9, 12, 15, 18, 21, 24]  # Your 8 time points
    
    # Generate temporal attention comparison
    plot_temporal_attention_comparison(retain_temporal, coi_temporal, time_points)
    
    # 2. Compute and visualize influence chains
    influence_data = compute_chain_influence(model_coi, test_data)
    plot_influence_chains(influence_data, feature_names, time_points)
    
    print(f"Figures saved to {output_dir}")
```

## Running the Visualization Generation

```bash
# Generate figures for paper
python analysis/model_analysis.py --model retain --generate-paper-figures
python analysis/model_analysis.py --model coi --generate-paper-figures
```

## Notes for Paper Submission

1. **Figure Quality**: Use `dpi=300` for high-resolution figures suitable for publication
2. **Size Constraints**: AAAI figures should be 0.48\textwidth (~3.5 inches) for side-by-side figures
3. **Font Size**: Use readable font sizes (10-12pt) that will be legible when printed
4. **Color**: Consider colorblind-friendly palettes and ensure figures work in grayscale
5. **File Format**: Save as PNG or PDF for best quality in LaTeX

## Performance Table Data

Based on your metrics files, here's the data for Table 1:

```python
# From your results
metrics_data = {
    'RETAIN': {'AUROC': 0.50, 'F1': 0.00, 'Accuracy': 0.90, 'Precision': 0.00},
    'CoI': {'AUROC': 0.75, 'F1': 0.667, 'Accuracy': 0.95, 'Precision': 1.00}
}

# Improvements
auroc_improvement = (0.75 - 0.50) / 0.50 * 100  # 50%
f1_improvement = (0.667 - 0.00) / 0.001 * 100   # ~67% (avoiding divide by zero)
```

Remember to load your actual feature names from `config/feature_names.yaml` when generating the visualizations! 