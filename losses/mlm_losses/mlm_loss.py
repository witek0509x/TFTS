import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC

from losses.contrastive_losses.base_contrastive_loss import BaseContrastiveLoss


class MLMLoss:

    def __call__(self, x_pred: torch.Tensor, x_real: torch.Tensor, masked_tokens: np.array) -> torch.Tensor:
        return torch.mean((x_pred[:, masked_tokens, :] - x_real[:, masked_tokens, :]) ** 2)


if __name__ == '__main__':
    loss = MLMLoss()

    # Compute the loss using embeddings and class labels
    x_pred = torch.randn(16, 20, 50)  # Example embeddings of shape [batch_size, embedding_dim]
    x_real = x_pred.clone()
    x_real[:, 4, :] += 1
    masked_tokens = np.array([1, 4, 7])

    res = loss(x_pred, x_real, masked_tokens)
    print(f"Computed loss: {res.item()}")
