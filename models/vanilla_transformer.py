import torch
from torch import nn
from pytorch_lightning import LightningModule


class TransformerModel(LightningModule):
    """
    Vanilla Transformer model for embedding time series using PyTorch Lightning.
    """

    def __init__(self, loss_fn, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, input_dim=50, lr=1e-4):
        """
        Args:
            loss_fn (BaseLoss): The loss function to be used in training.
            d_model (int): Dimensionality of the input sequence.
            nhead (int): Number of heads in multi-head attention.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Dimension of feedforward layers in the encoder.
            lr (float): Learning rate.
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Embedding layer for input tokens
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, d_model)  # Projection layer for embeddings
        self.loss_fn = loss_fn
        self.lr = lr

        self.train_loss_epoch = []
        self.val_loss_epoch = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the vanilla transformer model.

        Args:
            x (torch.Tensor): Input time series of shape [batch_size, seq_length, d_model].

        Returns:
            torch.Tensor: Output embeddings of shape [batch_size, d_model].
        """
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.linear(x[:, 0, :])  # Only return the embedding for the first token

    def training_step(self, batch, batch_idx):
        """
        Training step for the vanilla transformer model.

        Args:
            batch: Input batch of data.
            batch_idx: Index of the batch.

        Returns:
            Loss value for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the vanilla transformer model.

        Args:
            batch: Input batch of validation data.
            batch_idx: Index of the batch.

        Returns:
            Validation loss value for the batch.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the vanilla transformer model.

        Returns:
            torch.optim.Optimizer: Optimizer for model training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def save_model(self, path: str):
        """
        Save the model's state dictionary.

        Args:
            path (str): Path where the model should be saved.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        """
        Load the model's state dictionary.

        Args:
            path (str): Path from which to load the model.
        """
        self.load_state_dict(torch.load(path))

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        """
        # Print the average training loss for the epoch
        avg_train_loss = self.trainer.callback_metrics['train_loss'].mean().item()
        self.train_loss_epoch.append(avg_train_loss)
        print(f"\nEpoch {self.current_epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        """
        # Print the average validation loss for the epoch
        avg_val_loss = self.trainer.callback_metrics['val_loss'].mean().item()
        self.val_loss_epoch.append(avg_val_loss)
        print(f"Epoch {self.current_epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")
