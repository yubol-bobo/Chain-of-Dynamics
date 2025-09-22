#!/usr/bin/env python3
"""
StageNet: Stage-Aware Neural Networks for Health Risk Prediction

Implementation of StageNet model from:
"StageNet: Stage-Aware Neural Networks for Health Risk Prediction"

Key features:
1. Disease stage-aware modeling
2. Temporal stage transitions
3. Stage-specific feature learning
4. Adaptive stage attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StageAwareConv(nn.Module):
    """
    Stage-aware convolutional layer for learning stage-specific patterns.
    Uses different kernels to capture patterns at different disease stages.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_stages=4):
        super(StageAwareConv, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_stages = num_stages

        # Stage-specific convolutional layers
        self.stage_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
            for _ in range(num_stages)
        ])

        # Stage classification layer
        self.stage_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stages),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, seq_len, input_dim]
        Returns:
            stage_features: Stage-aware features [batch_size, seq_len, hidden_dim]
            stage_probs: Stage probabilities [batch_size, seq_len, num_stages]
        """
        batch_size, seq_len, input_dim = x.size()

        # Predict stage probabilities for each timestep
        stage_probs = self.stage_classifier(x)  # [batch_size, seq_len, num_stages]

        # Apply stage-specific convolutions
        x_transposed = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]

        stage_outputs = []
        for i, stage_conv in enumerate(self.stage_convs):
            stage_out = stage_conv(x_transposed)  # [batch_size, hidden_dim, seq_len]
            stage_out = stage_out.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
            stage_outputs.append(stage_out)

        # Weighted combination based on stage probabilities
        stage_features = torch.zeros(batch_size, seq_len, self.hidden_dim).to(x.device)
        for i, stage_out in enumerate(stage_outputs):
            stage_weight = stage_probs[:, :, i:i+1]  # [batch_size, seq_len, 1]
            stage_features += stage_weight * stage_out

        return stage_features, stage_probs


class TemporalStageTransition(nn.Module):
    """
    Models temporal transitions between disease stages.
    Uses LSTM to capture stage progression over time.
    """

    def __init__(self, stage_dim, hidden_dim, num_stages=4):
        super(TemporalStageTransition, self).__init__()

        self.stage_dim = stage_dim
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages

        # LSTM for stage transition modeling
        self.stage_lstm = nn.LSTM(
            input_size=stage_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        # Stage transition probability
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stages),
            nn.Softmax(dim=-1)
        )

    def forward(self, stage_features):
        """
        Args:
            stage_features: Stage features [batch_size, seq_len, stage_dim]
        Returns:
            transition_probs: Stage transition probabilities [batch_size, seq_len, num_stages]
            lstm_out: LSTM hidden states [batch_size, seq_len, hidden_dim]
        """
        # Model temporal dependencies
        lstm_out, _ = self.stage_lstm(stage_features)

        # Predict stage transitions
        transition_probs = self.transition_net(lstm_out)

        return transition_probs, lstm_out


class AdaptiveStageAttention(nn.Module):
    """
    Adaptive attention mechanism that considers disease stages.
    Combines temporal attention with stage-aware weighting.
    """

    def __init__(self, feature_dim, hidden_dim, num_stages=4):
        super(AdaptiveStageAttention, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages

        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Stage-aware attention
        self.stage_attention = nn.Sequential(
            nn.Linear(num_stages, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Combined attention
        self.attention_combine = nn.Linear(2, 1)

    def forward(self, features, stage_probs):
        """
        Args:
            features: Input features [batch_size, seq_len, feature_dim]
            stage_probs: Stage probabilities [batch_size, seq_len, num_stages]
        Returns:
            attended_features: Attended features [batch_size, feature_dim]
            attention_weights: Attention weights [batch_size, seq_len]
        """
        batch_size, seq_len, _ = features.size()

        # Temporal attention scores
        temporal_scores = self.temporal_attention(features).squeeze(-1)  # [batch_size, seq_len]

        # Stage-aware attention scores
        stage_scores = self.stage_attention(stage_probs).squeeze(-1)  # [batch_size, seq_len]

        # Combine attention scores
        combined_scores = torch.stack([temporal_scores, stage_scores], dim=-1)  # [batch_size, seq_len, 2]
        attention_logits = self.attention_combine(combined_scores).squeeze(-1)  # [batch_size, seq_len]

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_logits, dim=1)  # [batch_size, seq_len]

        # Apply attention to features
        attended_features = torch.sum(
            attention_weights.unsqueeze(-1) * features, dim=1
        )  # [batch_size, feature_dim]

        return attended_features, attention_weights


class StageNet(nn.Module):
    """
    StageNet model for stage-aware clinical prediction.

    Key components:
    1. Stage-aware convolutional layers
    2. Temporal stage transition modeling
    3. Adaptive stage attention
    4. Stage-specific feature learning
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_stages=4,
                 kernel_size=3, dropout=0.2):
        super(StageNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_stages = num_stages

        # Stage-aware convolution
        self.stage_conv = StageAwareConv(
            input_dim, hidden_dim, kernel_size, num_stages
        )

        # Temporal stage transition
        self.stage_transition = TemporalStageTransition(
            hidden_dim, hidden_dim, num_stages
        )

        # Adaptive stage attention
        self.stage_attention = AdaptiveStageAttention(
            hidden_dim, hidden_dim, num_stages
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Store attention weights for interpretability
        self.attention_weights = None
        self.stage_probs = None
        self.transition_probs = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass of StageNet model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            output: Prediction tensor of shape [batch_size, output_dim]
        """
        # Stage-aware feature extraction
        stage_features, stage_probs = self.stage_conv(x)

        # Temporal stage transition modeling
        transition_probs, lstm_features = self.stage_transition(stage_features)

        # Adaptive stage attention
        attended_features, attention_weights = self.stage_attention(
            lstm_features, transition_probs
        )

        # Store for interpretability
        self.attention_weights = attention_weights
        self.stage_probs = stage_probs
        self.transition_probs = transition_probs

        # Final prediction
        output = self.classifier(attended_features)

        return output

    def get_attention_weights(self):
        """
        Extract attention weights for interpretability.

        Returns:
            dict: Dictionary containing different types of attention weights
        """
        return {
            'temporal': self.attention_weights,      # Temporal attention weights
            'stage_probs': self.stage_probs,         # Stage probabilities
            'transition_probs': self.transition_probs # Stage transition probabilities
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