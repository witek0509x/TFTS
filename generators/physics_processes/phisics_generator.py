import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations

from generators.physics_processes.physics_process import physics_process_factory


class PhysicsProcessDataset(Dataset):
    """
    Dataset that generates time series from random physics generators.
    Each generator corresponds to a class, and subseries are extracted from the generated series.
    """

    def __init__(self, batch_size: int = 16, series_length: int = 1000, subseries_length: int = 50,
                 stride: int = 50, epoch_size: int = 1000, randomize=True, max_noise_factor=0.3):
        """
        Args:
            batch_size (int): The number of samples in each batch.
            series_length (int): The total length of each series.
            subseries_length (int): The length of each subseries extracted from the full series.
            stride (int): The stride to move the window for subseries extraction.
            num_labels (int): Number of unique labels (physics processes).
        """
        self.physics_processes = physics_process_factory(randomize, max_noise_factor)
        self.batch_size = batch_size
        self.series_length = series_length
        self.subseries_length = subseries_length
        self.stride = stride
        self.epoch_size = epoch_size
        self.randomize = randomize

        # Randomly assign generators to labels
        self.label_to_generator = {}

    def _generate_series(self, label):
        """
        Generate a series using the physics process corresponding to the given label.
        """
        if not self.randomize:
            generator = self.label_to_generator[label]
        else:
            generator = random.choice(self.physics_processes)

        full_series = generator.get_data(label, delta_t=1, N=self.series_length)
        subseries = [
            full_series[start:start + self.subseries_length]
            for start in range(0, self.series_length - self.subseries_length + 1, self.stride)
        ]
        return torch.Tensor(np.array(subseries).astype('float32'))

    def __len__(self):
        """
        Return the total number of batches in the dataset.
        """
        return self.epoch_size

    def __getitem__(self, index: int):
        """
        Get a batch of data ensuring at least two labels from the same process are present.

        Args:
            index (int): Index of the batch in the dataset.

        Returns:
            torch.Tensor: Batch of subseries as a tensor.
            torch.Tensor: Corresponding labels as a tensor.
        """
        # Generate data for the selected labels

        return self._generate_series(index), torch.tensor(index)

    def get_dataloader(self):
        """
        Return a PyTorch DataLoader for the dataset.

        Returns:
            DataLoader: Dataloader to iterate over the dataset in batches.
        """
        return DataLoader(self, batch_size=self.batch_size, shuffle=False, drop_last=True)


# Usage Example
if __name__ == "__main__":
    # Import previously defined physics processes
    # Create the dataset
    dataset = PhysicsProcessDataset(
        batch_size=16,
        series_length=1000,
        subseries_length=50,
        stride=50,
        epoch_size=16
    )

    # Create the dataloader
    dataloader = dataset.get_dataloader()

    # Iterate over batches
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Data shape: {data.shape}")  # Expected shape: [16, 50]
        print(f"Labels: {labels}")
        break  # Just printing the first batch
