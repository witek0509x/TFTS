from random import random

import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics.functional import r2_score

from losses.contrastive_losses.contrastive_loss_implementation import ContrastiveLoss
from models.positional_encoding import LearnablePositionalEncoding


class TransformerMLMModel(LightningModule):

    def __init__(self, loss_fn=ContrastiveLoss(margin=1.0), d_model=256, nhead=16, num_layers=8, dim_feedforward=1024, input_dim=1, output_dim=2, lr=1e-4):
        super(TransformerMLMModel, self).__init__()
        self.positional_encoding = LearnablePositionalEncoding(d_model, 1000)
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, output_dim)
        self.loss_fn = loss_fn
        self.lr = lr

        self.train_loss_epoch = []
        self.val_loss_epoch = []

        # For storing activations
        self.activations = {}

        # self._register_hooks()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return self.linear(x)

    def benchmark(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def training_step(self, batch, batch_idx):
        print()
        x, y = batch
        x_original = x.clone()
        masked_tokens = np.random.choice(range(x.shape[1]), int(x.shape[1] * 0.2))
        x[:, masked_tokens, :] = 0
        x_hat = self(x)
        mean_values = x_hat[:, masked_tokens, 0].mean()
        mean_values_var = x_hat[:, masked_tokens, 0].var()
        var_values = x_hat[:, masked_tokens, 0].mean()
        loss = -torch.distributions.Normal(x_hat[:, masked_tokens, 0][:, :, None], torch.exp(x_hat[:, masked_tokens, 1][:, :, None])).log_prob(
            x_original[:, masked_tokens, :]).mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("mean_values", mean_values, on_epoch=True, prog_bar=False)
        self.log("mean_values_var", mean_values_var, on_epoch=True, prog_bar=False)
        self.log("var_values", var_values, on_epoch=True, prog_bar=False)
        return loss

    def _register_hooks(self):
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().clone()

            return hook
        self.embedding.register_forward_hook(save_activation("embedding"))
        self.transformer_encoder.register_forward_hook(save_activation("transformer_encoder"))
        self.linear.register_forward_hook(save_activation("linear"))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_original = x.clone()
        masked_tokens = np.random.choice(range(x.shape[1]), int(x.shape[1] * 0.2))
        x[:, masked_tokens, :] = 0
        x_hat = self(x)
        mean_values = x_hat[:, masked_tokens, 0].mean()
        var_values = x_hat[:, masked_tokens, 0].mean()
        val_loss = -torch.distributions.Normal(x_hat[:, masked_tokens, 0][:, :, None], torch.exp(x_hat[:, masked_tokens, 1][:, :, None])).log_prob(
            x_original[:, masked_tokens, :]).mean()
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        self.log("validation_mean_values", mean_values, on_epoch=True, prog_bar=True)
        self.log("validation_var_values", var_values, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.o = optimizer
        return optimizer

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

    def on_train_epoch_end(self):
        # Print the average training loss for the epoch
        avg_train_loss = self.trainer.callback_metrics['train_loss'].mean().item()
        self.train_loss_epoch.append(avg_train_loss)
        print(f"\nEpoch {self.current_epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

    def on_validation_epoch_end(self):
        # Print the average validation loss for the epoch
        avg_val_loss = self.trainer.callback_metrics['val_loss'].mean().item()
        self.val_loss_epoch.append(avg_val_loss)
        print(f"Epoch {self.current_epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")

    def get_activations(self):
        return self.activations

if __name__ == "__main__":
    from generators.physics_processes.phisics_generator import PhysicsProcessDataset
    model = TransformerMLMModel()
    model.to('cuda')
    model.train()
    for (x, y) in PhysicsProcessDataset(
                                                batch_size=16,
                                                series_length=1000,
                                                subseries_length=50,
                                                stride=50,
                                                epoch_size=160
                                            ).get_dataloader():
        x = x.to('cuda')
        print(model.training_step((x, y), 0))