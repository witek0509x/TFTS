from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader

from generators.echo_state_generator import EchoStateNetwork
from generators.subseries_converter import EchoStateDataset

if __name__ == "__main__":
    # Define ESN parameters
    n_input = 1
    n_reservoir = 50
    spectral_radius = 0.9
    sparsity = 0.1
    input_scaling = 0.1
    leak_rate = 0.2
    device = "cuda"
    initial_seed = 42
    lr = 1e-4

    # Create ESN model
    esn = EchoStateNetwork(
        n_input=n_input,
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        device=device,
        random_state_torch=initial_seed
    )

    # Create dataset and dataloaders
    batch_size = 16
    series_length = 40
    tile_size = 1
    stride = 1
    padding = 0

    train_dataset = EchoStateDataset(
        num_series=10_000,
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
        initial_seed=initial_seed
    )
    val_dataset = EchoStateDataset(
        num_series=1_000,
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
        initial_seed=initial_seed
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="esn-{epoch:02d}-{val_loss:.2f}"
    )
    n_epoch_checkpoint = ModelCheckpoint(
        every_n_epochs=50,
        save_top_k=-1,
        filename="esn-every-n-epoch-{epoch:02d}"
    )

    # Define Lightning Trainer
    trainer = Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback, n_epoch_checkpoint]
    )

    # Train the model
    trainer.fit(esn, train_dataloaders=train_loader, val_dataloaders=val_loader)
