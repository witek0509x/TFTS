from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from generators.poisson_process_generator import PoissonProcessDataset
from losses.contrastive_losses.contrastive_loss_implementation import ContrastiveLoss
from models.vanilla_transformer import TransformerModel

# Main Training Loop
if __name__ == "__main__":
    # Define the custom loss function
    loss_function = ContrastiveLoss(margin=1.0)

    # Initialize the model with the loss function
    model = TransformerModel(loss_fn=loss_function)

    # Create the DataLoader for training and validation
    train_loader = PoissonProcessDataset(batch_size=16, series_length=1000, subseries_length=50, stride=50).get_dataloader()
    val_loader = PoissonProcessDataset(batch_size=16, series_length=1000, subseries_length=50, stride=50).get_dataloader()

    # Callbacks for checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min", filename="transformer-{epoch:02d}-{val_loss:.2f}"
    )

    # Create the PyTorch Lightning Trainer
    trainer = Trainer(max_epochs=200, callbacks=[checkpoint_callback])

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
