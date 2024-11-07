import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create learnable positional encoding matrix
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.uniform_(self.pe, -0.1, 0.1)  # Initialize the learnable parameters

    def forward(self, x):
        # Add learnable positional encoding
        x = x + self.pe[:x.size(1), :].reshape(1, x.size(1), -1)
        return x
