import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, Optional

from generators.echo_state_generator import EchoStateNetwork


class EchoStateDataset(Dataset):
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 num_series=100, series_length=100, tile_size=1, stride=1, padding=0,
                 n_input=1, n_reservoir=50, spectral_radius=0.4, sparsity=0.9, input_scaling=0.1,
                 leak_rate=0.9, device='cuda', initial_seed=0, esn_id=0, non_repeat=False, roll_every=1, training=True,
                 reset_every=-1, esn_store_state=False):
        """
        Dataset for Echo State Network generated data.
        
        Args:
            config: Configuration dictionary (optional)
            num_series: Number of series to generate
            series_length: Length of each series
            tile_size: Size of tiles for grouping data points
            stride: Stride between data points
            padding: Padding to add to the series
            n_input: Number of input dimensions
            n_reservoir: Size of the reservoir
            spectral_radius: Spectral radius of the reservoir
            sparsity: Sparsity of the reservoir
            input_scaling: Input scaling factor
            leak_rate: Leak rate of the reservoir
            device: Device to use for computations
            initial_seed: Initial random seed
            esn_id: ID for the ESN (for reproducibility)
            non_repeat: Whether to avoid repeating data
            roll_every: Frequency of input changes
        """
        # If config is provided, override default parameters
        if config is not None:
            dataset_config = config.get('dataset', {})
            esn_config = dataset_config.get('esn', {})
            
            # Dataset parameters could come from either train or val sections,
            # depending on how this instance is being used
            # We'll check for both but prioritize train
            train_config = dataset_config.get('train', {})
            val_config = dataset_config.get('val', {})
            
            # ESN parameters
            n_input = esn_config.get('n_input', n_input)
            n_reservoir = esn_config.get('n_reservoir', n_reservoir)
            spectral_radius = esn_config.get('spectral_radius', spectral_radius)
            sparsity = esn_config.get('sparsity', sparsity)
            input_scaling = esn_config.get('input_scaling', input_scaling)
            leak_rate = esn_config.get('leak_rate', leak_rate)
            esn_id = esn_config.get('esn_id', esn_id)
            roll_every = esn_config.get('roll_every', roll_every)
            esn_store_state = esn_config.get('esn_store_state', esn_store_state)
            
            # Dataset parameters - try train config first, then val config
            # This allows the caller to specify which type of dataset this is
            # by using the appropriate config section
            if training:
                num_series = train_config.get('num_series', num_series)
                series_length = train_config.get('series_length', series_length)
                tile_size = train_config.get('tile_size', tile_size)
                stride = train_config.get('stride', stride)
                padding = train_config.get('padding', padding)
                initial_seed = train_config.get('initial_seed', initial_seed)
                non_repeat = train_config.get('non_repeat', non_repeat)
                reset_every = train_config.get('reset_every', reset_every)
            elif val_config:
                num_series = val_config.get('num_series', num_series)
                series_length = val_config.get('series_length', series_length)
                tile_size = val_config.get('tile_size', tile_size)
                stride = val_config.get('stride', stride)
                padding = val_config.get('padding', padding)
                initial_seed = val_config.get('initial_seed', initial_seed)
                non_repeat = val_config.get('non_repeat', non_repeat)
                reset_every = val_config.get('reset_every', reset_every)
            
            # Get device from training config if available
            training_config = config.get('training', {})
            device = training_config.get('device', device)
        
        # Store parameters
        self.num_series = num_series
        self.series_length = series_length
        self.tile_size = tile_size
        self.stride = stride
        self.padding = padding
        self.initial_seed = initial_seed

        self.current_shift = 0
        self.items_provided = 0
        self.non_repeat = non_repeat
        self.roll_every = roll_every
        self.reset_every = reset_every
        
        # Store ESN parameters for logging/reproducibility
        self.esn_params = {
            'n_input': n_input,
            'n_reservoir': n_reservoir,
            'spectral_radius': spectral_radius,
            'sparsity': sparsity,
            'input_scaling': input_scaling,
            'leak_rate': leak_rate,
            'esn_id': esn_id,
            'roll_every': roll_every,
            'esn_store_state': esn_store_state,
        }
        
        # Store dataset parameters for logging/reproducibility
        self.dataset_params = {
            'num_series': num_series,
            'series_length': series_length,
            'tile_size': tile_size,
            'stride': stride,
            'padding': padding,
            'initial_seed': initial_seed,
            'non_repeat': non_repeat,
            'reset_every': reset_every
        }

        # Initialize ESN
        self.esn = EchoStateNetwork(
            n_input=n_input,
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            device=device,
            random_state_torch=esn_id,
            store_state=esn_store_state
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
        self.items_provided += 1
        if self.reset_every > 0:
            if self.items_provided > self.reset_every:
                self.items_provided = 0
                self.initial_seed += self.num_series
                print("-----------------------------------------------")
                print("Initial seed: {}".format(self.initial_seed))
                print("Number of series: {}".format(self.num_series))
                print("-----------------------------------------------")



        np.random.seed(self.initial_seed + self._get_next_shift(idx))
        self.esn.reset_state()
        series = self.esn.generate_series(self.series_length, roll_every=self.roll_every)

        pad_left = [series[0]] * self.padding
        pad_right = [series[-1]] * self.padding
        padded_series = pad_left + series + pad_right

        # Extract subseries with stride
        subseries = []
        for i in range(0, len(padded_series) - self.tile_size + 1, self.stride):
            subseries.append(padded_series[i:i + self.tile_size])

        return torch.Tensor(np.array(subseries)), torch.tensor(idx)
        
    def get_params(self):
        """Get all parameters for reproducibility."""
        return {
            'esn': self.esn_params,
            'dataset': self.dataset_params
        }

if __name__ == "__main__":
    # Example usage with config
    from utils.config_utils import load_config
    
    config = load_config("/home/wojciech/private/magisterka/TFTS/configs/transformer_mlm/final_experiment_config.yaml")
    
    # Create train and validation datasets
    train_dataset = EchoStateDataset(config)
    val_dataset = EchoStateDataset(config)
    
    # Sample data

    import matplotlib.pyplot as plt
    for i in range(20):
        x, y = train_dataset[i]
        plt.plot(x, alpha=0.3)
        print(f"Generated data shape: {x.shape}")
    plt.show()


