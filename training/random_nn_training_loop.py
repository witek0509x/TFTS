from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from generators.brownian_motion_generator import BrownianMotionDataset
from generators.poisson_process_generator import PoissonProcessDataset
from losses.contrastive_losses.contrastive_loss_implementation import ContrastiveLoss
from models.vanilla_transformer import TransformerModel
from pytorch_lightning.loggers import WandbLogger  # Import WandbLogger

import wandb  # Import wandb to initialize the project


# Main Training Loop
if __name__ == "__main__":
    # Initialize Wandb logger

    wandb.init(project="transformer_stochastic_processes")  # Optional: name your project

    # Define the Wandb logger
    wandb_logger = WandbLogger(project="transformer_stochastic_processes")

    # Define the custom loss function
    loss_function = ContrastiveLoss(margin=1.0)

    # Initialize the model with the loss function
    model = TransformerModel(loss_fn=loss_function)

    # Create the DataLoader for training and validation
    train_loader = BrownianMotionDataset(batch_size=16, series_length=1000, subseries_length=50, stride=50, intensity_count=20000, ranges={"means": (-20, 20), "std": (0.5, 10)}).get_dataloader()
    val_loader = BrownianMotionDataset(batch_size=16, series_length=1000, subseries_length=50, stride=50, ranges={"means": (-20, 20), "std": (0.5, 10)}).get_dataloader()

    # Callbacks for checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min", filename="transformer-{epoch:02d}-{val_loss:.2f}"
    )

    n_epoch_checkpoint = ModelCheckpoint(
        every_n_epochs=10,  # Replace 10 with the desired interval
        save_top_k=-1,  # Save all checkpoints for every n-th epoch
        filename="transformer-every-n-epoch-{epoch:02d}"
    )

    # Create the PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback, n_epoch_checkpoint],
        logger=wandb_logger
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Finish the wandb run (optional, but good practice)
    wandb.finish()