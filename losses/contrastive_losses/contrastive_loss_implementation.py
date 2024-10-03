import torch
import torch.nn.functional as F
from abc import ABC

from losses.contrastive_losses.base_contrastive_loss import BaseContrastiveLoss


class ContrastiveLoss(BaseContrastiveLoss):
    """
    Contrastive loss function where:
    - Embeddings from the same class are considered positive pairs (target: 0 distance).
    - Embeddings from different classes are considered negative pairs (target: margin distance).
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin (float): The margin for dissimilar pairs. Pairs with distance greater than this
                            margin will be penalized.
        """
        self.margin = margin

    def compute_loss(self, y_pred: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss for a batch of embeddings based on class labels.

        Args:
            y_pred (torch.Tensor): Embeddings of shape [batch_size, embedding_dim].
            classes (torch.Tensor): Class labels of shape [batch_size].

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        batch_size = y_pred.size(0)
        loss = 0.0

        # Compare all pairs of embeddings
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # Compute Euclidean distance between embeddings
                distance = F.pairwise_distance(y_pred[i].unsqueeze(0), y_pred[j].unsqueeze(0))

                # Same class -> positive pair (minimize distance)
                if classes[i] == classes[j]:
                    loss += distance.pow(2)  # minimize squared distance for same class

                # Different class -> negative pair (maximize distance with margin)
                else:
                    loss += F.relu(self.margin - distance).pow(2)  # maximize with margin for different class

        # Normalize loss by the number of pairs
        num_pairs = (batch_size * (batch_size - 1)) / 2
        loss = loss / num_pairs

        return loss


if __name__ == '__main__':
    # Define a contrastive loss function
    # Instantiate the loss with a margin of 1.0
    contrastive_loss_fn = ContrastiveLoss(margin=1.0)

    # Compute the loss using embeddings and class labels
    y_pred = torch.randn(10, 128)  # Example embeddings of shape [batch_size, embedding_dim]
    classes = torch.randint(0, 3, (10,))  # Example class labels with 3 different classes

    loss = contrastive_loss_fn(y_pred, classes)
    print(f"Computed loss: {loss.item()}")
