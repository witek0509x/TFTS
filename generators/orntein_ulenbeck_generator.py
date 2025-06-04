import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OrnsteinUhlenbeckDataset(Dataset):
    """
    Dataset that generates time series from Ornstein-Uhlenbeck processes with varying parameters.
    Each parameter set is treated as a class, and subseries are extracted from the generated series.
    """

    def __init__(self, batch_size: int = 16, num_of_positives=4, series_length: int = 1000, subseries_length: int = 50, stride: int = 50, parameter_count=100, ranges=None):
        """
        Args:
            batch_size (int): The number of samples in each batch.
            series_length (int): The total length of each Ornstein-Uhlenbeck process series.
            subseries_length (int): The length of each subseries extracted from the full series.
            stride (int): The stride to move the window for subseries extraction.
        """
        self.batch_size = batch_size
        self.series_length = series_length
        self.subseries_length = subseries_length
        self.stride = stride
        self.num_of_positives = num_of_positives

        # Randomly select 'parameter_count' different (theta, mu, sigma) triples
        if ranges is None:
            ranges = {
                "theta": (0.01, 1.0),
                "sigma": (0.1, 2.0)
            }
        self.thetas = np.random.uniform(*ranges["theta"], size=parameter_count)
        self.mus = np.zeros(parameter_count)
        self.sigmas = np.random.uniform(*ranges["sigma"], size=parameter_count)

    def get_parameters(self, labels):
        res = []
        for label in labels:
            res.append([self.thetas[label], self.mus[label], self.sigmas[label]])
        return torch.tensor(res)

    def _generate_ou_series(self, label):
        """
        Generate Ornstein-Uhlenbeck process for a given parameter set.
        """
        theta = self.thetas[label]
        mu = self.mus[label]
        sigma = self.sigmas[label]

        dt = 1.0  # Time step
        x = np.zeros(self.series_length)
        x[0] = np.random.normal(mu, sigma)
        for t in range(1, self.series_length):
            x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal()

        # Extract subseries using the given stride
        result = []
        for start in range(0, self.series_length - self.subseries_length + 1, self.stride):
            subseries = x[start:start + self.subseries_length]
            result.append(subseries)

        result = np.array(result)
        return result.astype('float32')

    def __len__(self):
        """
        Return the total number of subseries in the dataset.
        """
        return self.num_of_positives * len(self.thetas)

    def __getitem__(self, index: int):
        """
        Get a subseries and its corresponding class label.

        Args:
            index (int): Index of the subseries in the dataset.

        Returns:
            torch.Tensor: Subseries as a tensor.
            torch.Tensor: Corresponding class label as a tensor.
        """
        label = index % (len(self.thetas) * self.num_of_positives) // self.num_of_positives
        result = self._generate_ou_series(label)
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
    dataset = OrnsteinUhlenbeckDataset(batch_size=16, series_length=1000, subseries_length=50, stride=50)

    # Create the dataloader
    dataloader = dataset.get_dataloader()

    # Iterate over batches
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Data shape: {data.shape}")  # Expected shape: [16, 50]
        print(f"Labels: {labels}")
        break  # Just printing the first batch

