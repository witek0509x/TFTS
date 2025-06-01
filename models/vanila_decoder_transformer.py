from random import random
import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics.functional import r2_score
from typing import Dict, Any, Optional
import torch.nn.functional as F

from losses.contrastive_losses.contrastive_loss_implementation import ContrastiveLoss
from models.positional_encoding import LearnablePositionalEncoding


class TransformerDecoderModel(LightningModule):
    def __init__(self, config: Optional[Dict[str, Any]] = None, d_model=256, nhead=16,
                 num_layers=8, dim_feedforward=1024, input_dim=1, lr=1e-4):
        super().__init__()

        if config is not None:
            model_config = config.get('model', {})
            d_model = model_config.get('d_model', d_model)
            nhead = model_config.get('nhead', nhead)
            num_layers = model_config.get('num_layers', num_layers)
            dim_feedforward = model_config.get('dim_feedforward', dim_feedforward)
            input_dim = model_config.get('input_dim', input_dim)
            lr = model_config.get('lr', lr)

        self.config = config

        self.positional_encoding = LearnablePositionalEncoding(d_model, 1000)
        self.embedding = nn.Linear(input_dim, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, input_dim)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.input_dim = input_dim
        self.lr = lr

        self.train_loss_epoch = []
        self.val_loss_epoch = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (B, T, D)
        x = self.positional_encoding(x)  # (B, T, D)

        seq_len = x.size(1)

        # Use batch-first compatible TransformerDecoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # Dummy memory (not used in decoder-only setup)
        memory = torch.zeros(x.size(0), seq_len, self.d_model, device=x.device)

        x = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask  # This fixes the attention masking issue
        )

        return self.linear(x)  # (B, T, input_dim)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_input = x[:, :-1, :]
        y_target = x[:, 1:, :]

        y_hat = self(x_input)

        loss = F.mse_loss(y_hat, y_target)

        y_hat_flat = y_hat.flatten()
        y_target_flat = y_target.flatten()

        r2 = r2_score(y_hat_flat, y_target_flat)
        normalized_mse = loss / (torch.var(y_target_flat) + 1e-8)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_r2", r2, on_epoch=True, prog_bar=True)
        self.log("train_normalized_mse", normalized_mse, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        print(x.shape)
        x_input = x[:, :-1, :]
        y_target = x[:, 1:, :]

        y_hat = self(x_input)

        val_loss = F.mse_loss(y_hat, y_target)

        y_hat_flat = y_hat.flatten()
        y_target_flat = y_target.flatten()

        r2 = r2_score(y_hat_flat, y_target_flat)
        normalized_mse = val_loss / (torch.var(y_target_flat) + 1e-8)

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        self.log("val_r2", r2, on_epoch=True, prog_bar=True)
        self.log("val_normalized_mse", normalized_mse, on_epoch=True, prog_bar=False)

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

    def hyperparameters(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "input_dim": self.input_dim,
            "lr": self.lr,
        }

    def on_train_epoch_end(self):
        avg_train_loss = self.trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()
        self.train_loss_epoch.append(avg_train_loss)
        self.log("train_loss_epoch", avg_train_loss)

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.callback_metrics.get('val_loss', torch.tensor(0.0)).item()
        self.val_loss_epoch.append(avg_val_loss)
        self.log("val_loss_epoch", avg_val_loss)
