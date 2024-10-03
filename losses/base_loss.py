from abc import ABC, abstractmethod
import torch


class BaseLoss(ABC):
    """
    Abstract base class for loss functions.
    """

    @abstractmethod
    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss given predictions and true labels.

        Args:
            y_pred (torch.Tensor): Predictions from the model.
            y_true (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        pass

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Make the loss callable, so that it can be used like a regular function.
        """
        return self.compute_loss(y_pred, y_true)
