import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from generators.brownian_motion_generator import BrownianMotionDataset
from generators.orntein_ulenbeck_generator import OrnsteinUhlenbeckDataset

class GeneratorWrapperDataset:
    """
    Wrapper dataset class that parses a YAML configuration and initializes the appropriate generator.
    """

    def __init__(self, config_path, mode="train"):
        """
        Initialize the wrapper dataset by parsing the YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
            mode (str): Mode of the dataset, either "train" or "val".
        """
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        dataset_config = config.get("dataset", {}).get(mode, {})
        generator_type = dataset_config.get("generator_type")
        generator_params = dataset_config.get("generator_params", {})

        if generator_type == "BrownianMotion":
            self.dataset = BrownianMotionDataset(**generator_params)
        elif generator_type == "OrnsteinUhlenbeck":
            self.dataset = OrnsteinUhlenbeckDataset(**generator_params)
        else:
            raise ValueError(f"Unsupported generator type: {generator_type}")

    def get_dataloader(self):
        """
        Return the DataLoader from the initialized dataset.

        Returns:
            DataLoader: Dataloader to iterate over the dataset in batches.
        """
        return self.dataset.get_dataloader()

# Usage Example
if __name__ == "__main__":
    # Example configuration file path
    config_path = "/home/wojciech/private/magisterka/TFTS/configs/stochastic/brownian_motion_config.yaml"

    # Create the wrapper dataset for training
    train_dataset = GeneratorWrapperDataset(config_path, mode="train")
    train_dataloader = train_dataset.get_dataloader()

    # Visualize a single batch of trajectories
    for batch_idx, (data, labels) in enumerate(train_dataloader):
        plt.figure(figsize=(10, 6))
        unique_labels = labels.unique()
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label.item(): color for label, color in zip(unique_labels, colors)}

        for trajectory, label in zip(data, labels):
            plt.plot(trajectory.flatten().numpy(), color=label_to_color[label.item()], alpha=0.3, label=f"Class {label.item()}")
        plt.show()
        break