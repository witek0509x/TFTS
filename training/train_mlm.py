from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from generators.physics_processes.phisics_generator import PhysicsProcessDataset
from models.vanila_mlm_transformer import TransformerMLMModel

if __name__ == "__main__":
    # wandb.init(project="transformer_stochastic_processes")
    # wandb_logger = WandbLogger(project="transformer_stochastic_processes", name="physics_first")
    model = TransformerMLMModel(lr=1e-30)

    train_loader = PhysicsProcessDataset(
        batch_size=16,
        series_length=1000,
        subseries_length=50,
        stride=50,
        epoch_size=10_000
    ).get_dataloader()
    val_loader = PhysicsProcessDataset(
        batch_size=16,
        series_length=1000,
        subseries_length=50,
        stride=50,
        epoch_size=1_000
    ).get_dataloader()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min", filename="transformer-{epoch:02d}-{val_loss:.2f}"
    )
    n_epoch_checkpoint = ModelCheckpoint(
        every_n_epochs=50,
        save_top_k=-1,
        filename="transformer-every-n-epoch-{epoch:02d}"
    )

    trainer = Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback, n_epoch_checkpoint],
        # logger=wandb_logger
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # wandb.finish()