import torch
from torch.utils.data import Dataset
import numpy as np

from generators.echo_state_generator import EchoStateNetwork


class EchoStateDataset(Dataset):
    def __init__(self, num_series, series_length, tile_size, stride, padding,
                 n_input, n_reservoir, spectral_radius, sparsity, input_scaling,
                 leak_rate, device, initial_seed=0, esn_id=0, non_repeat=False, roll_every=1):
        self.num_series = num_series
        self.series_length = series_length
        self.tile_size = tile_size
        self.stride = stride
        self.padding = padding
        self.initial_seed = initial_seed

        self.current_shift = 0
        self.non_repeat = non_repeat
        self.roll_every = roll_every

        self.esn = EchoStateNetwork(
            n_input=n_input,
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            device=device,
            random_state_torch=esn_id
        )

    def __len__(self):
        return self.num_series

    def _get_next_shift(self, idx):
        if self.non_repeat:
            self.current_shift += 1
            return self.current_shift
        else:
            return idx

    def __getitem__(self, idx):
        np.random.seed(self.initial_seed + self._get_next_shift(idx))
        self.esn.reset_state()
        series = self.esn.generate_series(self.series_length, roll_every=self.roll_every)

        pad_left = [series[0]] * self.padding
        pad_right = [series[-1]] * self.padding
        padded_series = pad_left + series + pad_right

        num_tiles = (len(padded_series) - self.tile_size) // self.stride + 1
        tiles = [padded_series[i * self.stride: i * self.stride + self.tile_size] for i in range(num_tiles)]

        return torch.tensor(tiles, dtype=torch.float32), torch.tensor([self.initial_seed + self._get_next_shift(idx)])


import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Dataset and DataLoader parameters
    num_series = int(100)
    series_length = 1000
    tile_size = 10
    stride = 10
    padding = 0
    batch_size = 16

    # ESN parameters
    n_input = 1
    n_reservoir = 10
    spectral_radius = 0.9
    sparsity = 0.1
    input_scaling = 0.1
    leak_rate = 0.2
    device = 'cuda'
    initial_seed = 42

    # Create dataset
    dataset = EchoStateDataset(
        num_series=num_series,
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
        initial_seed=initial_seed,
        non_repeat=True
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Fetch one batch
    for batch in dataloader:
        x, y = batch
        break  # Take only the first batch

    # Validate shape
    print(f"Series batch shape: {x.shape}")
    print(f"Labels: {y}")

