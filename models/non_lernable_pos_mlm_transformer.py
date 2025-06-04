from random import random

import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torchmetrics.functional import r2_score
from typing import Dict, Any, Optional, Tuple

from generators.subseries_converter import EchoStateDataset
from losses.contrastive_losses.contrastive_loss_implementation import ContrastiveLoss
from models.cosine_pos_encoding import CosinePositionalEncoding
from models.positional_encoding import LearnablePositionalEncoding


class TransformerMLMModelV2(LightningModule):

    def __init__(self, config: Optional[Dict[str, Any]] = None, loss_fn=None, d_model=256, nhead=16,
                 num_layers=8, dim_feedforward=1024, input_dim=1, lr=1e-4, masking_ratio=0.2):
        """
        Transformer model with masked language modeling for time series.

        Args:
            config: Configuration dictionary (optional)
            loss_fn: Loss function
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of the feedforward network
            input_dim: Input dimension
            lr: Learning rate
            masking_ratio: Ratio of tokens to mask during training
        """
        super(TransformerMLMModelV2, self).__init__()

        # If config is provided, override default parameters
        if config is not None:
            model_config = config.get('model', {})
            d_model = model_config.get('d_model', d_model)
            nhead = model_config.get('nhead', nhead)
            num_layers = model_config.get('num_layers', num_layers)
            dim_feedforward = model_config.get('dim_feedforward', dim_feedforward)
            input_dim = model_config.get('input_dim', input_dim) + 1
            lr = model_config.get('lr', lr)
            masking_ratio = model_config.get('masking_ratio', masking_ratio)

        # Store the configuration
        self.config = config

        # Initialize model components
        self.positional_encoding = LearnablePositionalEncoding(d_model, 1000)
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, input_dim-1)

        # Set the loss function
        if loss_fn is None:
            self.loss_fn = ContrastiveLoss(margin=1.0)
        else:
            self.loss_fn = loss_fn

        # Model hyperparameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.input_dim = input_dim
        self.lr = lr
        self.masking_ratio = masking_ratio

        # Training metrics
        self.train_loss_epoch = []
        self.val_loss_epoch = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return self.linear(x)

    def forward_with_embbeding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return self.linear(x), x

    def benchmark(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_original = x.clone()

        # Apply masking based on masking_ratio
        masked_tokens = np.random.choice(range(x.shape[1]), int(x.shape[1] * self.masking_ratio))
        x[:, masked_tokens, :] = 0
        mask_col = torch.zeros(x.shape[1])
        mask_col[masked_tokens] = 1
        mask_col = mask_col.unsqueeze(0).expand(x.shape[0], -1).unsqueeze(-1).to('cuda')

        x = torch.cat([x, mask_col], dim=-1)
        x_hat = self(x)

        # Calculate metrics
        variance = torch.mean((x_hat[:, masked_tokens, :] - torch.mean(x_original)) ** 2)
        mse = torch.mean((x_hat[:, masked_tokens, :] - x_original[:, masked_tokens, :]) ** 2)
        normalized_variance = torch.var(x_original[:, masked_tokens, :]) - torch.minimum(variance, torch.var(
            x_original[:, masked_tokens, :]))
        r2 = r2_score(x_hat[:, masked_tokens, :].flatten(), x_original[:, masked_tokens, :].flatten())

        # Compute loss
        loss = mse  # + normalized_variance * 0.1

        # Log metrics to wandb
        self.log("train_r2", r2, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_variance", variance, on_epoch=True, prog_bar=False)
        self.log("train_normalized_variance", normalized_variance, on_epoch=True, prog_bar=False)
        self.log("train_mse", mse, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_original = x.clone()

        # Apply masking based on masking_ratio
        masked_tokens = np.random.choice(range(x.shape[1]), int(x.shape[1] * self.masking_ratio))
        x[:, masked_tokens, :] = 0
        mask_col = torch.zeros(x.shape[1])
        mask_col[masked_tokens] = 1
        mask_col = mask_col.unsqueeze(0).expand(x.shape[0], -1).unsqueeze(-1).to('cuda')
        x = torch.cat([x, mask_col], dim=-1)

        # Forward pass with masked input
        x_hat = self(x)

        # Calculate metrics
        variance = torch.mean((x_hat[:, masked_tokens, :] - torch.mean(x_original)) ** 2)
        mse = torch.mean((x_hat[:, masked_tokens, :] - x_original[:, masked_tokens, :]) ** 2)
        normalized_variance = torch.var(x_original[:, masked_tokens, :]) - torch.minimum(variance, torch.var(
            x_original[:, masked_tokens, :]))

        # Compute loss and R2 score
        val_loss = mse  # + normalized_variance * 0.1
        val_r2 = r2_score(x_hat[:, masked_tokens, :].flatten(), x_original[:, masked_tokens, :].flatten())

        # Log metrics to wandb
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        self.log("val_r2", val_r2, on_epoch=True, prog_bar=True)
        self.log("val_variance", variance, on_epoch=True, prog_bar=False)
        self.log("val_normalized_variance", normalized_variance, on_epoch=True, prog_bar=False)
        self.log("val_mse", mse, on_epoch=True, prog_bar=False)

        return val_loss

    def configure_optimizers(self):
        print(self.lr, type(self.lr))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.o = optimizer
        return optimizer

    def save_model(self, path: str):
        """Save model to a file."""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        """Load model from a file."""
        self.load_state_dict(torch.load(path))

    def hyperparameters(self) -> Dict[str, Any]:
        """Return model hyperparameters for logging."""
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "input_dim": self.input_dim,
            "lr": self.lr,
            "masking_ratio": self.masking_ratio
        }

    def on_train_epoch_end(self):
        # Calculate and log average training loss for the epoch
        avg_train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()
        self.train_loss_epoch.append(avg_train_loss)
        self.log("train_loss_epoch", avg_train_loss)
        print(f"\nEpoch {self.current_epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

    def on_validation_epoch_end(self):
        # Calculate and log average validation loss for the epoch
        avg_val_loss = self.trainer.callback_metrics.get('val_loss', torch.tensor(0.0)).item()
        self.val_loss_epoch.append(avg_val_loss)
        self.log("val_loss_epoch", avg_val_loss)
        print(f"Epoch {self.current_epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")

    def get_activations(self):
        return self.activations


if __name__ == "__main__":
    # Example usage with config
    from utils.config_utils import load_config

    config = load_config("/home/wojciech/private/magisterka/TFTS/configs/transformer_mlm/non_lernable_config.yaml")
    model = TransformerMLMModelV2(config=config)
    model.to('cuda')
    model.train()

    # Create datasets
    train_dataset = EchoStateDataset(config=config)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for (x, y) in train_loader:
        x = x.to('cuda')
        loss = model.training_step((x, y), 0)
        raise ""
        print(f"Training loss: {loss.item()}")