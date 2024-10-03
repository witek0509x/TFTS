import abc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from probes.probe import Probe


class LinearProbe(Probe):
    """
    A linear probe that trains a simple linear model on top of the frozen embeddings
    from the transformer model to predict continuous labels.
    """

    def __init__(self, model_path: str, input_dim: int, output_dim: int = 1, lr: float = 1e-3, benchmark=False):
        """
        Args:
            model_path (str): Path to the trained model checkpoint.
            input_dim (int): Dimensionality of the transformer model's embeddings.
            output_dim (int): Number of output dimensions (default: 1 for regression).
            lr (float): Learning rate for the linear probe.
        """
        super().__init__(model_path)
        self.linear = nn.Linear(input_dim, output_dim)  # Linear model
        self.criterion = nn.MSELoss()  # Assuming regression
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)
        self.load_model()  # Load the trained transformer model
        self.linear.to(self.model.device)
        self.benchmark = benchmark

    def finetune(self, dataloader: DataLoader, num_epochs: int = 10):
        """
        Fine-tunes the linear probe on top of frozen transformer embeddings.

        Args:
            dataloader (DataLoader): DataLoader for the training data.
            num_epochs (int): Number of epochs to fine-tune for.
        """
        self.model.eval()  # Make sure transformer model is frozen
        self.linear.train()
        print("Probe device:", next(self.linear.parameters()).device)
        print("Transformer device:", next(self.model.parameters()).device)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.float().to('cuda')
                y = dataloader.dataset.get_parameters(y)
                y = y.float().to('cuda')
                with torch.no_grad():  # Transformer is frozen
                    if not self.benchmark:
                        embeddings = self.model(x)
                    else:
                        embeddings = self.model.benchmark(x)
                # Pass embeddings through linear model
                preds = self.linear(embeddings)
                loss = self.criterion(preds, y)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss / len(dataloader):.4f}")

    def probe_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Probes the model with a single input data point.

        Args:
            x (torch.Tensor): Input time series.

        Returns:
            torch.Tensor: Model's prediction for the input data.
        """
        self.model.eval()
        self.linear.eval()

        with torch.no_grad():
            embedding = self.model(x.unsqueeze(0))  # Add batch dimension
            output = self.linear(embedding)

        return output

    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluates the probe on a set of datapoints.

        Args:
            dataloader (DataLoader): DataLoader for the validation data.

        Returns:
            float: Mean Squared Error (MSE) on the validation set.
        """
        self.model.eval()
        self.linear.eval()

        preds_list = []
        labels_list = []

        total_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.float().to('cuda')
                y = dataloader.dataset.get_parameters(y)
                y = y.float().to('cuda')
                if not self.benchmark:
                    embeddings = self.model(x)
                else:
                    embeddings = self.model.benchmark(x)
                preds = self.linear(embeddings)
                loss = self.criterion(preds, y)
                total_loss += loss.item()
                preds_list.append(preds.cpu().numpy())
                labels_list.append(y.cpu().numpy())

        preds = np.concatenate(preds_list)
        y_val = np.concatenate(labels_list)

        avg_loss = total_loss / len(dataloader)
        print(f"Validation MSE: {avg_loss:.4f}")
        return avg_loss, preds, y_val
