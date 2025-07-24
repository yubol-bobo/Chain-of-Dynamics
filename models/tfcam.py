#!/usr/bin/env python3
"""
TFCAM (Time-Feature Cross Attention Mechnism) Model

This module implements the TFCAM model with DyT (Dynamic Tanh) normalization,
which combines temporal attention and cross-feature attention to detect
how feature A at time t affects feature B at time t+k.

Key Components:
1. DyT (Dynamic Tanh) normalization layer
2. Multi-head self-attention for cross-feature interactions
3. Temporal and feature-level attention mechanisms
4. Comprehensive interpretability tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class DyT(nn.Module):
    """
    Dynamic Tanh (DyT) normalization layer.
    
    This layer applies a learnable tanh transformation with dynamic scaling
    and shifting parameters, providing adaptive normalization capabilities.
    """
    def __init__(self, num_features, alpha_init_value=0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Apply the dynamic tanh function element-wise
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module for cross-feature interactions"""
    
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.emb_dim)
        output = self.out_proj(context)
        return output, attn_weights


class CrossFeatureAttention(nn.Module):
    """
    Module for capturing dependencies between different features across time.
    Uses DyT normalization instead of LayerNorm for adaptive normalization.
    """
    
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(CrossFeatureAttention, self).__init__()
        self.self_attention = MultiHeadSelfAttention(emb_dim, num_heads, dropout)
        # Replace LayerNorm with DyT
        self.norm1 = DyT(emb_dim)
        self.norm2 = DyT(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_dim, emb_dim)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and DyT normalization
        attn_out, attn_weights = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward with residual connection and DyT normalization
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class TFCAM(nn.Module):
    """
    Time-Feature Cross Attention Networks (TFCAM)
    
    A model combining temporal attention and cross-feature attention,
    designed to detect how feature A at time t affects feature B at time t+k.
    
    Key Features:
    - Temporal attention for visit-level importance
    - Feature-level attention for variable importance
    - Cross-feature transformer layers for complex interactions
    - DyT normalization for adaptive scaling
    - Comprehensive interpretability tools
    """
    
    def __init__(self, input_dim, emb_dim, hidden_dim, num_heads, num_layers, 
                 output_dim=1, dropout=0.2, max_seq_len=50):
        super(TFCAM, self).__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, emb_dim)
        
        # Temporal attention
        self.temporal_lstm = nn.LSTM(
            emb_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.temporal_attn = nn.Linear(hidden_dim * 2, 1)
        
        # Feature-level attention
        self.feature_lstm = nn.LSTM(
            emb_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.feature_attn = nn.Linear(hidden_dim * 2, emb_dim)
        
        # Cross-feature transformer layers
        self.cross_attn_layers = nn.ModuleList([
            CrossFeatureAttention(emb_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, output_dim)
        )
        
        # Positional encoding
        self.register_buffer(
            "pos_encoding", 
            self._get_positional_encoding(max_seq_len, emb_dim)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for interpretability
        self.temporal_weights = None
        self.feature_weights = None
        self.cross_attn_weights = []
        
    def _get_positional_encoding(self, max_seq_len, emb_dim):
        """Generate positional encoding for transformer"""
        pe = torch.zeros(max_seq_len, emb_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x):
        """
        Forward pass of TFCAM model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            output: Prediction tensor of shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Clear previous attention weights
        self.cross_attn_weights = []
        
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, emb_dim]
        
        # Add positional encoding (ensure device match)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        
        # Temporal attention
        temporal_lstm_out, _ = self.temporal_lstm(x)
        temporal_attn_weights = torch.sigmoid(self.temporal_attn(temporal_lstm_out))
        self.temporal_weights = temporal_attn_weights
        
        # Feature-level attention
        feature_lstm_out, _ = self.feature_lstm(x)
        feature_attn_weights = torch.sigmoid(self.feature_attn(feature_lstm_out))
        self.feature_weights = feature_attn_weights
        
        # Apply feature attention
        x = x * feature_attn_weights
        
        # Cross-feature attention layers
        for layer in self.cross_attn_layers:
            x, attn_weights = layer(x)
            self.cross_attn_weights.append(attn_weights)
        
        # Apply temporal attention
        x = x * temporal_attn_weights
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # [batch_size, emb_dim]
        
        # Output projection
        output = self.output_proj(x)
        
        return output
    
    def get_attention_weights(self):
        """
        Get all attention weights for interpretability.
        Returns:
            dict: Dictionary containing temporal, feature, and cross-feature attention weights
        """
        return {
            'temporal': self.temporal_weights,
            'feature': self.feature_weights,
            'cross_feature': self.cross_attn_weights
        }
    
    def compute_contributions(self, patient_idx=0):
        """
        Compute feature contributions for a specific patient.
        
        Args:
            patient_idx: Index of the patient in the batch
            
        Returns:
            np.ndarray: Contribution matrix of shape [seq_len, input_dim]
        """
        if self.temporal_weights is None or self.feature_weights is None:
            raise ValueError("Model must be run forward first to compute contributions")
        
        # Get attention weights for the specific patient
        a_t = self.temporal_weights[patient_idx].squeeze(-1)  # [seq_len]
        b_t = self.feature_weights[patient_idx]  # [seq_len, emb_dim]
        
        # Get embedding weights
        W_emb = self.embedding.weight  # [emb_dim, input_dim]
        
        # Get the original input for this patient
        # Note: This requires access to the original input tensor
        # For now, we'll compute a simplified version
        seq_len = a_t.size(0)
        input_dim = self.input_dim
        
        contributions = torch.zeros(seq_len, input_dim)
        
        # Compute contributions for each time step and feature
        for t in range(seq_len):
            for j in range(input_dim):
                contributions[t, j] = a_t[t] * torch.sum(b_t[t] * W_emb[:, j])
        
        return np.array(contributions.cpu().tolist())
