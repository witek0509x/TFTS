from abc import ABC, abstractmethod
import torch


class BaseContrastiveLoss(ABC):
    """
    Abstract base class for loss functions.
    """

    @abstractmethod
    def compute_loss(self, y_pred: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss given predictions and true labels.

        Args:
            y_pred (torch.Tensor): Predictions from the model.
            classes (torch.Tensor): Classes to produce positive and negative pairs.

        Returns:
            torch.Tensor: Computed loss.
        """
        pass

    def __call__(self, y_pred: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """
        Make the loss callable, so that it can be used like a regular function.
        """
        return self.compute_loss(y_pred, classes)
