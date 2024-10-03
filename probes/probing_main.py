from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from generators.poisson_process_generator import PoissonProcessDataset
from models.vanilla_transformer import TransformerModel
from probes.linear_probe import LinearProbe
import matplotlib.pyplot as plt

# Probe Evaluation Loop
if __name__ == "__main__":
    # Path to the pre-trained model checkpoint (from previous training loop)
    checkpoint_path = "/home/wojciech/private/magisterka/TFTS/training/lightning_logs/version_8/checkpoints/transformer-epoch=154-val_loss=0.06.ckpt"

    # Load the trained model
    model = TransformerModel.load_from_checkpoint(checkpoint_path)

    # Create the DataLoader for evaluation
    val_loader = PoissonProcessDataset(batch_size=16, series_length=1000, subseries_length=50, stride=50).get_dataloader()

    # Initialize the linear probe with the transformer model
    probe = LinearProbe(model_path=checkpoint_path, input_dim=128)  # Assuming `d_model` is 128
    # Fine-tune the linear probe using the training data (optional)
    # If needed, create a train_loader like val_loader and call finetune
    train_loader = PoissonProcessDataset(batch_size=16, series_length=1000, subseries_length=50, stride=50, intensity_count=100).get_dataloader()
    probe.finetune(train_loader)

    # Evaluate the probe on the validation dataset
    val_loss, predictions, labels = probe.evaluate(val_loader)
    plt.scatter(labels, predictions, alpha=0.5)
    print(f"Validation Loss (MSE): {val_loss:.4f}")
