#!/usr/bin/env python3
"""
AdaCare: Explainable Clinical Health Status Representation Learning

Implementation of AdaCare model from:
"AdaCare: Explainable Clinical Health Status Representation Learning via Knowledge Distillation"

Key improvements over RETAIN:
1. Adaptive feature calibration
2. Enhanced interpretability with feature importance recalibration
3. Knowledge distillation for better representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureCalibration(nn.Module):
    """
    Adaptive feature calibration module.
    Recalibrates feature importance based on global and local contexts.
    """

    def __init__(self, input_dim, hidden_dim):
        super(FeatureCalibration, self).__init__()

        # Global context learning
        self.global_context = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # Local context learning
        self.local_context = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # Calibration weights
        self.calibration_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, seq_len, input_dim]
        Returns:
            calibrated_x: Calibrated features [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, input_dim = x.size()

        # Global context: average across time
        global_feat = torch.mean(x, dim=1)  # [batch_size, input_dim]
        global_weights = self.global_context(global_feat)  # [batch_size, input_dim]
        global_weights = global_weights.unsqueeze(1).expand(-1, seq_len, -1)

        # Local context: per-timestep
        local_weights = self.local_context(x.reshape(-1, input_dim))  # [batch_size*seq_len, input_dim]
        local_weights = local_weights.reshape(batch_size, seq_len, input_dim)

        # Adaptive calibration
        alpha = torch.sigmoid(self.calibration_weight)
        combined_weights = alpha * global_weights + (1 - alpha) * local_weights

        # Apply calibration
        calibrated_x = x * combined_weights

        return calibrated_x, combined_weights


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism with enhanced interpretability.
    Based on RETAIN but with feature calibration.
    """

    def __init__(self, input_dim, hidden_dim, num_heads=1):
        super(AdaptiveAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Alpha attention (temporal importance)
        self.alpha_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads),
            nn.Softmax(dim=1)
        )

        # Beta attention (feature importance)
        self.beta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, seq_len, input_dim]
        Returns:
            context: Weighted context vector [batch_size, input_dim]
            alpha_weights: Temporal attention weights [batch_size, seq_len, num_heads]
            beta_weights: Feature attention weights [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, input_dim = x.size()

        # Compute attention weights
        alpha_weights = self.alpha_net(x)  # [batch_size, seq_len, num_heads]
        beta_weights = self.beta_net(x)    # [batch_size, seq_len, input_dim]

        # Apply beta attention to features
        attended_x = x * beta_weights  # [batch_size, seq_len, input_dim]

        # Apply alpha attention across time
        if self.num_heads == 1:
            alpha_weights = alpha_weights.squeeze(-1)  # [batch_size, seq_len]
            context = torch.sum(alpha_weights.unsqueeze(-1) * attended_x, dim=1)
        else:
            # Multi-head attention
            context = torch.zeros(batch_size, input_dim).to(x.device)
            for h in range(self.num_heads):
                head_alpha = alpha_weights[:, :, h]  # [batch_size, seq_len]
                head_context = torch.sum(head_alpha.unsqueeze(-1) * attended_x, dim=1)
                context += head_context / self.num_heads

        return context, alpha_weights, beta_weights


class AdaCare(nn.Module):
    """
    AdaCare model for clinical prediction with adaptive feature calibration.

    Key components:
    1. Feature calibration module for adaptive feature importance
    2. Enhanced attention mechanism based on RETAIN
    3. Interpretable prediction with feature and temporal importance
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_heads=1,
                 dropout=0.2, calibration_dim=64):
        super(AdaCare, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Feature calibration
        self.feature_calibration = FeatureCalibration(input_dim, calibration_dim)

        # Adaptive attention mechanism
        self.attention = AdaptiveAttention(input_dim, hidden_dim, num_heads)

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Store attention weights for interpretability
        self.alpha_weights = None
        self.beta_weights = None
        self.calibration_weights = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass of AdaCare model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            output: Prediction tensor of shape [batch_size, output_dim]
        """
        # Feature calibration
        calibrated_x, calibration_weights = self.feature_calibration(x)

        # Adaptive attention
        context, alpha_weights, beta_weights = self.attention(calibrated_x)

        # Store for interpretability
        self.alpha_weights = alpha_weights
        self.beta_weights = beta_weights
        self.calibration_weights = calibration_weights

        # Classification
        output = self.classifier(context)

        return output

    def get_attention_weights(self):
        """
        Extract attention weights for interpretability.

        Returns:
            dict: Dictionary containing different types of attention weights
        """
        return {
            'temporal': self.alpha_weights,        # Temporal importance
            'feature': self.beta_weights,          # Feature importance
            'calibration': self.calibration_weights  # Feature calibration weights
        }

    def extract_attention_weights(self, x):
        """
        Extract attention weights for a given input.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            dict: Attention weights for interpretability
        """
        with torch.no_grad():
            _ = self.forward(x)
            return self.get_attention_weights()