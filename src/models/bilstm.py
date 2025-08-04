#!/usr/bin/env python3
"""
Bidirectional LSTM Baseline Model

A simple bidirectional LSTM baseline for temporal sequence modeling in healthcare data.
This serves as a strong baseline for comparison with more sophisticated attention-based models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM baseline model for clinical time series prediction.
    
    This model provides a strong baseline using bidirectional LSTM layers
    with dropout regularization and a simple classification head.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1, dropout=0.2):
        super(BiLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        """
        Forward pass of BiLSTM model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            output: Prediction tensor of shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the final hidden state (concatenated forward and backward)
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        final_output = lstm_out[:, -1, :]  # Take last timestep
        
        # Classification
        output = self.classifier(final_output)
        
        return output
    
    def get_attention_weights(self):
        """
        Dummy method for compatibility with attention-based models.
        BiLSTM doesn't have attention weights.
        """
        return {
            'temporal': None,
            'feature': None,
            'cross_feature': None
        } 