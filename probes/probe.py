import abc
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vanilla_transformer import TransformerModel


class Probe(abc.ABC):
    """
    Abstract base class for probes to evaluate the model's learned embeddings.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): Path to the trained model checkpoint.
        """
        self.model_path = model_path
        self.model = None  # The model will be loaded later

    @abc.abstractmethod
    def finetune(self, dataloader: DataLoader, num_epochs: int = 10):
        """
        Fine-tunes the probe on the provided dataset.

        Args:
            dataloader (DataLoader): DataLoader for the training data.
            num_epochs (int): Number of epochs to fine-tune for.
        """
        pass

    @abc.abstractmethod
    def probe_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Probes the model with a single input data point.

        Args:
            x (torch.Tensor): Input time series.

        Returns:
            torch.Tensor: Model's prediction for the input data.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Evaluates the probe on a set of datapoints.

        Args:
            dataloader (DataLoader): DataLoader for the validation data.

        Returns:
            float: Accuracy of the probe on the validation set.
        """
        pass

    def load_model(self):
        """
        Load the trained transformer model from the checkpoint.
        """
        self.model = TransformerModel.load_from_checkpoint(self.model_path)
        self.model.eval()  # Set model to evaluation mode
