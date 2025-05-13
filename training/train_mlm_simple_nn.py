import argparse
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from generators.physics_processes.phisics_generator import PhysicsProcessDataset
from generators.subseries_converter import EchoStateDataset
from models.trivial_nn import TrivialNN
from models.vanila_mlm_transformer import TransformerMLMModel
from utils.config_utils import (
    load_config, 
    initialize_wandb, 
    setup_model_checkpoint_callback, 
    setup_wandb_logger, 
    log_hyperparameters,
    log_model_to_wandb
)


def train(config_path, resume=False, run_id=None):
    """
    Train a TransformerMLMModel using configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        resume: Whether to resume training from a checkpoint
        run_id: wandb run ID to resume
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize wandb
    initialize_wandb(config, resume=resume, id=run_id)
    wandb_logger = setup_wandb_logger(config)
    
    # Create model
    model = TrivialNN(config=config)
    
    # Log hyperparameters
    log_hyperparameters(model, config)
    
    # Setup data
    dataset_config = config.get('dataset', {})
    train_config = dataset_config.get('train', {})
    val_config = dataset_config.get('val', {})
    
    # Create datasets
    train_dataset = EchoStateDataset(config=config, training=True)
    val_dataset = EchoStateDataset(config=config, training=False)
    
    # Get batch sizes from config
    train_batch_size = train_config.get('batch_size', 16)
    val_batch_size = val_config.get('batch_size', 16)
    
    # Get shuffle settings from config
    train_shuffle = train_config.get('shuffle', True)
    val_shuffle = val_config.get('shuffle', False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=val_shuffle)
    
    # Set up model checkpoints
    checkpoint_callbacks = setup_model_checkpoint_callback(config)
    checkpoint_callback = checkpoint_callbacks['checkpoint']
    n_epoch_checkpoint = checkpoint_callbacks['n_epoch_checkpoint']
    
    # Get training settings
    training_config = config.get('training', {})
    max_epochs = training_config.get('max_epochs', 100000)
    
    # Set up trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, n_epoch_checkpoint],
        logger=wandb_logger
    )
    
    # Train model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Log best model to wandb
    if checkpoint_callback.best_model_path:
        log_model_to_wandb(
            checkpoint_callback.best_model_path, 
            aliases=["best", f"epoch_{trainer.current_epoch}"]
        )
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TransformerMLMModel")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/transformer_mlm/default.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume training from a checkpoint"
    )
    parser.add_argument(
        "--run_id", 
        type=str, 
        default=None,
        help="wandb run ID to resume"
    )
    
    args = parser.parse_args()
    train(args.config, args.resume, args.run_id)