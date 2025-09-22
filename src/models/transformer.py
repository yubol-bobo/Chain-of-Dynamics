#!/usr/bin/env python3
"""
Transformer Baseline Model

A transformer-based baseline model for temporal sequence modeling in healthcare data.
This serves as a modern attention-based baseline using standard transformer architecture
with positional encoding and multi-head self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    Adds sinusoidal positional encodings to input embeddings.
    """

    def __init__(self, emb_dim, max_seq_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, emb_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer so it's not considered a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape [seq_len, batch_size, emb_dim]

        Returns:
            x: Input with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer baseline model for clinical time series prediction.

    This model uses standard transformer encoder architecture with:
    - Multi-head self-attention
    - Positional encoding
    - Feed-forward networks
    - Layer normalization and residual connections
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, num_heads=8, num_layers=6,
                 output_dim=1, dropout=0.2, max_seq_len=50):
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, emb_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(emb_dim, max_seq_len, dropout)

        # Transformer encoder layers with stability improvements
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Additional layer normalization for stability
        self.layer_norm = nn.LayerNorm(emb_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Global average pooling for sequence aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Store attention weights for interpretability
        self.attention_weights = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for better stability
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Initialize transformer layers with smaller weights
                for param in module.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param, gain=0.1)

    def forward(self, x):
        """
        Forward pass of Transformer model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            output: Prediction tensor of shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.size()

        # Input embedding
        x = self.input_embedding(x)  # [batch_size, seq_len, emb_dim]

        # Add positional encoding (need to transpose for pos_encoder)
        x = x.transpose(0, 1)  # [seq_len, batch_size, emb_dim]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, emb_dim]

        # Create attention mask (optional - can be used for padding)
        # For now, we assume no padding mask is needed
        attn_mask = None

        # Transformer encoder
        transformer_out = self.transformer_encoder(x, mask=attn_mask)
        # transformer_out: [batch_size, seq_len, emb_dim]

        # Apply layer normalization for stability
        transformer_out = self.layer_norm(transformer_out)

        # Global average pooling across sequence dimension
        pooled = transformer_out.transpose(1, 2)  # [batch_size, emb_dim, seq_len]
        pooled = self.global_pool(pooled).squeeze(-1)  # [batch_size, emb_dim]

        # Classification
        output = self.classifier(pooled)

        return output

    def get_attention_weights(self):
        """
        Extract attention weights for interpretability.
        Note: This is a simplified version. For full attention analysis,
        you would need to modify the transformer layers to return attention weights.
        """
        # This is a placeholder - in practice, you'd need to modify
        # the transformer encoder to return attention weights
        return {
            'temporal': None,  # Would contain temporal attention weights
            'feature': None,   # Would contain feature attention weights
            'cross_feature': None  # Not directly applicable to standard transformer
        }

    def extract_attention_weights(self, x):
        """
        Extract attention weights during forward pass.
        This requires modifying the forward pass to capture attention weights.
        """
        # This would require a custom implementation to capture
        # attention weights from each transformer layer
        pass


class TransformerWithCLS(TransformerModel):
    """
    Transformer variant with CLS token (similar to BERT).
    Uses a special classification token for sequence-level prediction.
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, num_heads=8, num_layers=6,
                 output_dim=1, dropout=0.2, max_seq_len=50):
        super(TransformerWithCLS, self).__init__(
            input_dim, emb_dim, hidden_dim, num_heads, num_layers,
            output_dim, dropout, max_seq_len + 1  # +1 for CLS token
        )

        # CLS token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(self, x):
        """
        Forward pass with CLS token.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            output: Prediction tensor of shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.size()

        # Input embedding
        x = self.input_embedding(x)  # [batch_size, seq_len, emb_dim]

        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, emb_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, seq_len+1, emb_dim]

        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len+1, batch_size, emb_dim]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len+1, emb_dim]

        # Transformer encoder
        transformer_out = self.transformer_encoder(x)
        # transformer_out: [batch_size, seq_len+1, emb_dim]

        # Use CLS token output for classification
        cls_output = transformer_out[:, 0, :]  # [batch_size, emb_dim]

        # Classification
        output = self.classifier(cls_output)

        return output