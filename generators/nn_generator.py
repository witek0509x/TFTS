import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.random_dynamics import RandomDynamics


class NNGenerator(Dataset):
    """
    Dataset that generates time series from Poisson processes with varying intensities.
    Each intensity is treated as a class, and subseries are extracted from the generated series.
    """

    def __init__(self, batch_size: int = 16, num_of_positives=4, series_length: int = 1000, subseries_length: int = 50, stride: int = 50, store_params=False):
        """
        Args:
            batch_size (int): The number of samples in each batch.
            series_length (int): The total length of each Poisson process series.
            subseries_length (int): The length of each subseries extracted from the full series.
            stride (int): The stride to move the window for subseries extraction.
        """
        self.batch_size = batch_size
        self.series_length = series_length
        self.subseries_length = subseries_length
        self.stride = stride
        self.num_of_positives = num_of_positives

        self.lags = 2
        self.noise_size = 0
        self.network = RandomDynamics(4, self.lags).to('cuda')
        self.iterations = 0

        self.params = []
        self.store_params = store_params

    def _generate_dynamics(self):
        """
        Generate Poisson processes for each of the 100 intensities, extract subseries, and assign class labels.
        """
        # Generate Poisson process with given intensity
        if self.iterations % self.num_of_positives == 0:
            if self.store_params:
                self.params.append(self.network.to('cpu'))
                self.network = RandomDynamics(4, self.lags).to('cuda')
            self.network.reset()
        state = torch.normal(0, torch.ones(self.lags)).to("cuda")
        states = []
        with torch.no_grad():
            for i in range(self.series_length):
                output = self.network(state).clone()
                # print(output)
                states.append(output)
                state = torch.normal(0, torch.ones(self.lags)*3).to("cuda")
                state[0] = output.item()
        full_series = torch.concat(states)
        # Extract subseries using the given stride
        result = []
        for start in range(0, self.series_length - self.subseries_length + 1, self.stride):
            subseries = full_series[start:start + self.subseries_length]
            result.append(subseries)
        result = torch.stack(result)
        return result.float()

    def get_parameters(self, label):
        return self.params[label]

    def __len__(self):
        """
        Return the total number of subseries in the dataset.
        """
        return 1000

    def __getitem__(self, index: int):
        """
        Get a subseries and its corresponding class label.

        Args:
            index (int): Index of the subseries in the dataset.

        Returns:
            torch.Tensor: Subseries as a tensor.
            torch.Tensor: Corresponding class label as a tensor.
        """
        result = self._generate_dynamics()
        label = self.iterations // self.num_of_positives
        self.iterations += 1
        return result, label

    def get_dataloader(self):
        """
        Return a PyTorch DataLoader for the dataset.

        Returns:
            DataLoader: Dataloader to iterate over the dataset in batches.
        """
        return DataLoader(self, batch_size=self.batch_size, shuffle=False, drop_last=True)


# Usage Example
if __name__ == "__main__":
    # Create the dataset
    dataset = NNGenerator(batch_size=16, series_length=100, subseries_length=1, stride=1)

    # Create the dataloader
    dataloader = dataset.get_dataloader()

    # Iterate over batches
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Data shape: {data.shape}")  # Expected shape: [16, 50]
        print(f"Labels: {labels}")
        break  # Just printing the first batch
