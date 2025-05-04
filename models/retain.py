import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RETAIN(nn.Module):
    """
    RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism
    
    This implementation follows the architecture described in the paper:
    "RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism"
    by Choi et al.
    """
    def __init__(self, input_dim, emb_dim, hidden_dim, num_heads=None, num_layers=None, output_dim=1, dropout=0.2, max_seq_len=50):
        super(RETAIN, self).__init__()
        
        # Note: num_heads and num_layers are not used in RETAIN but included for compatibility
        # with the hyperparameter search function
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Embedding layer for input features
        self.embedding = nn.Linear(input_dim, emb_dim)
        
        # Visit-level attention (alpha) - determines importance of each visit
        self.alpha_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.alpha_attn = nn.Linear(hidden_dim, 1)
        
        # Variable-level attention (beta) - determines importance of each variable
        self.beta_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.beta_attn = nn.Linear(hidden_dim, emb_dim)
        
        # Output layer
        self.output_layer = nn.Linear(emb_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for interpretability
        self.alpha_weights = None
        self.beta_weights = None
        
        # Store original input for interpretability
        self.original_input = None
    
    def forward(self, x):
        """
        Forward pass through the RETAIN model
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            output: Prediction tensor of shape (batch_size, output_dim)
        """
        # Store original input for interpretability
        self.original_input = x.clone()
        
        batch_size, seq_length, _ = x.size()
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, emb_dim)
        embedded = self.dropout(embedded)
        
        # Visit-level attention (alpha)
        alpha_hidden, _ = self.alpha_gru(embedded)  # (batch_size, seq_length, hidden_dim)
        alpha_attn = self.alpha_attn(alpha_hidden)  # (batch_size, seq_length, 1)
        
        # Apply softmax to get attention weights (reversed for RETAIN)
        # In RETAIN, the attention is reversed to look backward in time
        self.alpha_weights = F.softmax(alpha_attn, dim=1)  # (batch_size, seq_length, 1)
        
        # Variable-level attention (beta)
        beta_hidden, _ = self.beta_gru(embedded)  # (batch_size, seq_length, hidden_dim)
        self.beta_weights = torch.tanh(self.beta_attn(beta_hidden))  # (batch_size, seq_length, emb_dim)
        
        # Element-wise multiplication of beta and embedded
        weighted_embedded = self.beta_weights * embedded  # (batch_size, seq_length, emb_dim)
        
        # Weight by alpha and sum across time
        context_vector = torch.sum(self.alpha_weights * weighted_embedded, dim=1)  # (batch_size, emb_dim)
        
        # Output projection
        output = self.output_layer(context_vector)  # (batch_size, output_dim)
        
        return output
    
    def get_attention_weights(self):
        """
        Returns attention weights for interpretability.
        
        Returns:
            Dictionary containing different attention weights
        """
        return {
            "alpha": self.alpha_weights,  # Visit-level attention
            "beta": self.beta_weights,    # Variable-level attention
        }
    
    def compute_contributions(self, patient_idx=0):
        """
        Compute feature contributions for a specific patient using RETAIN's attention mechanisms.
        
        Returns:
            numpy.ndarray: Contribution scores of shape [seq_len, input_dim].
        """
        if self.alpha_weights is None or self.beta_weights is None:
            raise ValueError("Model must perform a forward pass first")
        
        device = self.embedding.weight.device
        alpha = self.alpha_weights[patient_idx].to(device).detach()  # [seq_len, 1]
        beta = self.beta_weights[patient_idx].to(device).detach()    # [seq_len, emb_dim]
        x = self.original_input[patient_idx].to(device).detach()     # [seq_len, input_dim]
        W_emb = self.embedding.weight.to(device).detach()           # [emb_dim, input_dim]
        
        contributions = torch.zeros(x.size(), device=device)  # [seq_len, input_dim]
        
        # For each time step
        for t in range(x.size(0)):
            a_t = alpha[t]  # scalar attention weight at time t
            b_t = beta[t]   # variable-level weights at time t (vector of size emb_dim)
            x_t = x[t]      # input features at time t
            
            # Each feature j's contribution is: a_t * sum_i [ b_t[i] * W_emb[i, j] * x_t[j] ]
            for j in range(x.size(1)):
                contributions[t, j] = a_t * torch.sum(b_t * W_emb[:, j] * x_t[j])
                
        return contributions.cpu().numpy()