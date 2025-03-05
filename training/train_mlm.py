from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from generators.physics_processes.phisics_generator import PhysicsProcessDataset
from generators.subseries_converter import EchoStateDataset
from models.vanila_mlm_transformer import TransformerMLMModel

def train():
    wandb.init(project="magisterka")
    wandb_logger = WandbLogger(project="magisterka", name="ESN_first_experiment")
    model = TransformerMLMModel(lr=1e-5)

    n_input = 1
    n_reservoir = 10
    spectral_radius = 0.99
    sparsity = 0.1
    input_scaling = 0.1
    leak_rate = 0.7
    device = 'cuda'
    train_len = 1000
    val_len = 10
    lr = 1e-4

    ESN_id = 40

    # Create dataset and dataloaders
    batch_size = 128
    series_length = 100
    tile_size = 1
    stride = 1
    padding = 0

    train_dataset = EchoStateDataset(
        num_series=train_len,
        series_length=series_length,
        tile_size=tile_size,
        stride=stride,
        padding=padding,
        n_input=n_input,
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        device=device,
        initial_seed=val_len,
        esn_id=ESN_id,
        non_repeat = True
    )
    val_dataset = EchoStateDataset(
        num_series=val_len,
        series_length=series_length,
        tile_size=tile_size,
        stride=stride,
        padding=padding,
        n_input=n_input,
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        device=device,
        esn_id=ESN_id,
        initial_seed=0,
        non_repeat = False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min", filename="transformer-{epoch:02d}-{val_loss:.6f}"
    )
    n_epoch_checkpoint = ModelCheckpoint(
        every_n_epochs=100,
        save_top_k=-1,
        filename="transformer-every-n-epoch-{epoch:06d}"
    )

    trainer = Trainer(
        max_epochs=100000,
        callbacks=[checkpoint_callback, n_epoch_checkpoint],
        logger=wandb_logger
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    wandb.finish()