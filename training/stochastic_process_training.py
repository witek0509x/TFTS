import argparse
import os

from pytorch_lightning import Trainer
import torch
import wandb

from generators.generator_wrapper_dataset import GeneratorWrapperDataset
from utils.config_utils import (
    load_config,
    initialize_wandb,
    setup_model_checkpoint_callback,
    setup_wandb_logger,
    log_hyperparameters,
    log_model_to_wandb
)
from models.vanilla_transformer import TransformerModel

def download_model_and_config(run_id):
    api = wandb.Api()
    run = api.run(f"stochastic/{run_id}")
    config = run.config
    artifact = api.artifact(f"stochastic/model-{run_id}:best")
    model_dir = artifact.download()
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
    if not model_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    model_path = os.path.join(model_dir, model_files[0])
    return model_path, config

def load_model(model_path, config):
    model = TransformerModel(
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        input_dim=config['model']['input_dim'],
        lr=config['model']['lr']
    )
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
        model = TransformerModel(
            **config['model']
        )

    log_hyperparameters(model, config)

    train_dataset = GeneratorWrapperDataset(config_path, mode="train")
    val_dataset = GeneratorWrapperDataset(config_path, mode="val")

    train_loader = train_dataset.get_dataloader()
    val_loader = val_dataset.get_dataloader()

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
