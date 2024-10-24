import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PoissonProcessDataset(Dataset):
    """
    Dataset that generates time series from Poisson processes with varying intensities.
    Each intensity is treated as a class, and subseries are extracted from the generated series.
    """

    def __init__(self, batch_size: int = 16, num_of_positives=4, series_length: int = 1000, subseries_length: int = 50, stride: int = 50, intensity_count=100):
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

        # Randomly select 100 different intensities from the range [1, 100]
        self.intensities = np.random.uniform(0, 1, size=intensity_count)
        self.data = []
        self.labels = []

    def get_parameters(self, labels):
        res = []
        for label in labels:
            res.append(self.intensities[label])
        return torch.tensor(res)

    def _generate_poisson_series(self, label):
        """
        Generate Poisson processes for each of the 100 intensities, extract subseries, and assign class labels.
        """
        # Generate Poisson process with given intensity
        full_series = np.random.poisson(lam=self.intensities[label], size=self.series_length)
        full_series = np.cumsum(full_series)
        # Extract subseries using the given stride
        result = []
        for start in range(0, self.series_length - self.subseries_length + 1, self.stride):
            subseries = full_series[start:start + self.subseries_length]
            result.append(subseries)

        # Convert data and labels to numpy arrays for efficient indexing
        result = np.array(result)
        return result.astype('float32')

    def __len__(self):
        """
        Return the total number of subseries in the dataset.
        """
        return self.num_of_positives * len(self.intensities)

    def __getitem__(self, index: int):
        """
        Get a subseries and its corresponding class label.

        Args:
            index (int): Index of the subseries in the dataset.

        Returns:
            torch.Tensor: Subseries as a tensor.
            torch.Tensor: Corresponding class label as a tensor.
        """
        label = index % (len(self.intensities) * self.num_of_positives) // self.num_of_positives
        result = self._generate_poisson_series(label)
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
    dataset = PoissonProcessDataset(batch_size=16, series_length=1000, subseries_length=50, stride=50)

    # Create the dataloader
    dataloader = dataset.get_dataloader()

    # Iterate over batches
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Data shape: {data.shape}")  # Expected shape: [16, 50]
        print(f"Labels: {labels}")
        break  # Just printing the first batch
