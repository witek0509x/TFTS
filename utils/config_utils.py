import os
import yaml
from typing import Dict, Any, Optional, Union
import wandb


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def initialize_wandb(config: Dict[str, Any], resume: bool = False, id: Optional[str] = None) -> wandb.run:
    """
    Initialize Weights & Biases with the given configuration.
    
    Args:
        config: Configuration dictionary
        resume: Whether to resume a previous run
        id: Optional run ID to resume
        
    Returns:
        Initialized wandb run
    """
    experiment_config = config.get('experiment', {})
    
    # Extract experiment settings
    project = experiment_config.get('project', 'magisterka')
    name = experiment_config.get('name', 'default_experiment')
    tags = experiment_config.get('tags', [])
    
    # Initialize wandb
    run = wandb.init(
        project=project,
        name=name,
        tags=tags,
        config=config,
        resume=resume,
        id=id
    )
    
    return run


def setup_model_checkpoint_callback(config: Dict[str, Any], log_to_wandb: bool = True) -> Dict[str, Any]:
    """
    Set up model checkpoint callbacks based on configuration.
    
    Args:
        config: Configuration dictionary
        log_to_wandb: Whether to log checkpoints to wandb
        
    Returns:
        Dictionary of checkpoint callbacks
    """
    from pytorch_lightning.callbacks import ModelCheckpoint
    
    training_config = config.get('training', {})
    checkpoint_config = training_config.get('checkpoint', {})
    n_epoch_checkpoint_config = training_config.get('n_epoch_checkpoint', {})
    
    # Main checkpoint for best models
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        save_top_k=checkpoint_config.get('save_top_k', 1),
        mode=checkpoint_config.get('mode', 'min'),
        filename=checkpoint_config.get('filename', 'transformer-{epoch:02d}-{val_loss:.6f}')
    )
    
    # Additional checkpoint every N epochs
    n_epoch_checkpoint = ModelCheckpoint(
        every_n_epochs=n_epoch_checkpoint_config.get('every_n_epochs', 200),
        save_top_k=n_epoch_checkpoint_config.get('save_top_k', -1),
        filename=n_epoch_checkpoint_config.get('filename', 'transformer-every-n-epoch-{epoch:06d}')
    )
    
    return {
        'checkpoint': checkpoint_callback,
        'n_epoch_checkpoint': n_epoch_checkpoint
    }


def setup_wandb_logger(config: Dict[str, Any]) -> wandb.wandb_sdk.wandb_run.Run:
    """
    Set up Weights & Biases logger.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        WandbLogger instance
    """
    from pytorch_lightning.loggers import WandbLogger
    
    experiment_config = config.get('experiment', {})
    project = experiment_config.get('project', 'magisterka')
    name = experiment_config.get('name', 'default_experiment')
    log_model = experiment_config.get('log_model', True)
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=project,
        name=name,
        log_model=log_model
    )
    
    return wandb_logger


def log_hyperparameters(model, config: Dict[str, Any], run=None) -> None:
    """
    Log model hyperparameters to wandb.
    
    Args:
        model: The model instance
        config: Configuration dictionary
        run: Optional wandb run instance
    """
    if run is None:
        run = wandb.run
        
    if run is None:
        raise ValueError("No active wandb run found. Initialize wandb first.")
    
    # Log all config parameters
    run.config.update(config, allow_val_change=True)
    
    # If the model has hyperparameters attribute or method, log those too
    if hasattr(model, 'hyperparameters'):
        if callable(model.hyperparameters):
            hyperparams = model.hyperparameters()
        else:
            hyperparams = model.hyperparameters
        run.config.update(hyperparams, allow_val_change=True)


def log_model_to_wandb(model_path: str, aliases: Optional[list] = None) -> None:
    """
    Log a saved model file to wandb.
    
    Args:
        model_path: Path to the saved model file
        aliases: Optional list of aliases for the model
    """
    if wandb.run is None:
        raise ValueError("No active wandb run found. Initialize wandb first.")
    
    wandb.save(model_path, base_path=os.path.dirname(model_path))
    
    if aliases:
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}", 
            type="model",
            description="Trained model checkpoint"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=aliases)


if __name__ == "__main__":
    # Example usage:
    config = load_config("configs/transformer_mlm/default.yaml")
    print("Loaded configuration:", config) 