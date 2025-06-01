import argparse
import os

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch
import wandb

from generators.subseries_converter import EchoStateDataset
from utils.config_utils import (
    load_config,
    initialize_wandb,
    setup_model_checkpoint_callback,
    setup_wandb_logger,
    log_hyperparameters,
    log_model_to_wandb
)
from models.vanila_decoder_transformer import TransformerDecoderModel

def download_model_and_config(run_id):
    api = wandb.Api()
    run = api.run(f"mlm_esn/{run_id}")
    config = run.config
    artifact = api.artifact(f"mlm_esn/model-{run_id}:best")
    model_dir = artifact.download()
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
    if not model_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    model_path = os.path.join(model_dir, model_files[0])
    return model_path, config

def load_model(model_path, config):
    model = TransformerDecoderModel(config=config)
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def train(config_path=None, resume=False, run_id=None):
    config = load_config(config_path)
    initialize_wandb(config)
    wandb_logger = setup_wandb_logger(config)

    if resume:
        model_path, config_d = download_model_and_config(run_id)
        model = load_model(model_path, config_d if config is None else config)
    else:
        model = TransformerDecoderModel(config=config)

    log_hyperparameters(model, config)

    dataset_config = config.get('dataset', {})
    train_dataset = EchoStateDataset(config=config, training=True)
    val_dataset = EchoStateDataset(config=config, training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.get('train', {}).get('batch_size', 16),
        shuffle=dataset_config.get('train', {}).get('shuffle', True)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_config.get('val', {}).get('batch_size', 16),
        shuffle=dataset_config.get('val', {}).get('shuffle', False)
    )

    checkpoint_callbacks = setup_model_checkpoint_callback(config)
    checkpoint_callback = checkpoint_callbacks['checkpoint']
    n_epoch_checkpoint = checkpoint_callbacks['n_epoch_checkpoint']

    trainer = Trainer(
        max_epochs=config.get('training', {}).get('max_epochs', 100000),
        callbacks=[checkpoint_callback, n_epoch_checkpoint],
        logger=wandb_logger
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if checkpoint_callback.best_model_path:
        log_model_to_wandb(
            checkpoint_callback.best_model_path,
            aliases=["best", f"epoch_{trainer.current_epoch}"]
        )

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transformer_mlm/default.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()
    train(args.config, args.resume, args.run_id)
