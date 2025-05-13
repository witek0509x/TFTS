import math
import torch
import torch.nn as nn

class CosinePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(CosinePositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a long enough positional encoding matrix using cosine functions.
        # Each position i gets a vector of dimension d_model computed as:
        #   pe(i, j) = cos( i / (10000^(j/d_model)) )
        # Here we use the same denominator term as in the standard sinusoidal formulation.
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.cos(position * div_term)  # Shape: (max_len, d_model)

        # Register as buffer to avoid updating during training.
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of the same shape with positional encoding added.
        """
        seq_len = x.size(1)
        # Add the positional encoding to x (broadcasting over the batch dimension)
        x = x + self.pe[:, :seq_len]
        return x
