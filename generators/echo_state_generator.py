import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class EchoStateNetwork(nn.Module):
    def __init__(self, n_input, n_reservoir, spectral_radius=0.95, sparsity=0.1,
                 input_scaling=1.0, leak_rate=1.0, device='cpu', random_state_torch=42, random_state_numpy=42):
        """
        Initialize the Echo State Network (ESN).

        Parameters:
        - n_input (int): Dimensionality of the input vector.
        - n_reservoir (int): Number of neurons in the reservoir.
        - spectral_radius (float): Desired spectral radius of the reservoir.
        - sparsity (float): Proportion of non-zero connections in the reservoir.
        - input_scaling (float): Scaling factor for input weights.
        - leak_rate (float): Leak rate for the leaky integrator.
        - device (str): Device to run the ESN on ('cpu' or 'cuda').
        - random_state (int or None): Seed for reproducibility.
        """
        super(EchoStateNetwork, self).__init__()
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.device = device


        torch.manual_seed(random_state_torch)
        np.random.seed(random_state_numpy)

        # Initialize weight matrices
        self.W_in = self._initialize_input_weights().to(self.device)
        self.W = self._initialize_reservoir().to(self.device)
        self.W_out = self._initialize_output_weights().to(self.device)

        # Initialize reservoir state
        self.reset_state()

        # List to store hidden states
        self.hidden_states = []

    def _initialize_input_weights(self):
        """
        Initialize the input weight matrix W_in with uniform distribution.
        """
        W_in = torch.rand(self.n_reservoir, self.n_input) * 2 - 1  # Uniform between -1 and 1
        W_in *= self.input_scaling
        return W_in

    def _initialize_reservoir(self):
        """
        Initialize the reservoir weight matrix W with specified sparsity and spectral radius.
        """
        # Initialize a sparse random matrix
        W = torch.rand(self.n_reservoir, self.n_reservoir) * 2 - 1  # Uniform between -1 and 1
        mask = torch.rand_like(W) > self.sparsity
        W[mask] = 0

        # Compute spectral radius
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(W).cpu().numpy()
            current_spectral_radius = max(abs(eigenvalues))
            if current_spectral_radius == 0:
                raise ValueError("Spectral radius of the reservoir is zero. Adjust sparsity or initialization.")
            W *= self.spectral_radius / current_spectral_radius.real

        return W

    def _initialize_output_weights(self):
        """
        Initialize the output weight matrix W_out randomly.
        Maps [reservoir_state; input] to single output.
        """
        # The output is a single number, so W_out has shape (1, n_reservoir + n_input)
        W_out = torch.rand(1, self.n_reservoir) * 2 - 1  # Uniform between -1 and 1
        return W_out

    def reset_state(self):
        """
        Reset the reservoir state to zero.
        """
        self.state = torch.zeros(self.n_reservoir, 1).to(self.device)
        self.hidden_states = []

    def forward(self, input_vector):
        """
        Perform one inference step with the given input vector.

        Parameters:
        - input_vector (torch.Tensor): Input vector of shape (n_input,).

        Returns:
        - output (float): The output number.
        """
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, dtype=torch.float32)

        if input_vector.shape[0] != self.n_input:
            raise ValueError(f"Input vector must have shape ({self.n_input},), got {input_vector.shape}")

        input_vector = input_vector.view(-1, 1).to(self.device)  # Shape: (n_input, 1)

        # Update reservoir state
        pre_activation = torch.matmul(self.W_in, input_vector) + torch.matmul(self.W, self.state)
        updated_state = (1 - self.leak_rate) * self.state + self.leak_rate * torch.tanh(pre_activation)
        self.state = updated_state

        # Concatenate reservoir state and input for output
        # concatenated = torch.cat((self.state, input_vector), dim=0)  # Shape: (n_reservoir + n_input, 1)

        # Compute output
        y = torch.matmul(self.W_out, self.state)  # Shape: (1, 1)
        output = y.item()  # Extract scalar

        # Store hidden state
        self.hidden_states.append(self.state.detach().cpu().numpy())

        return output

    def generate_series(self, steps, input_dimensionality=None, roll_every=1):
        """
        Generate a series of outputs by feeding random inputs.

        Parameters:
        - steps (int): Number of steps to generate.
        - input_dimensionality (int or None): If specified, overrides the ESN's n_input for generating random inputs.

        Returns:
        - outputs (list of float): Generated output series.
        """
        outputs = []
        input_vector = None
        for step in range(steps):
            # Generate a random input vector
            if input_vector is None or step % roll_every == 0:
                if input_dimensionality is not None:
                    input_vector = np.random.randn(input_dimensionality)
                else:
                    input_vector = np.random.randn(self.n_input)
                    print(input_vector)
            # Perform a step
            y = self.forward(input_vector)
            outputs.append(y)
        return outputs

    def get_hidden_states(self):
        """
        Retrieve the list of hidden (reservoir) states.

        Returns:
        - hidden_states (list of np.ndarray): List of reservoir states.
        """
        return self.hidden_states

if __name__ == "__main__":
    n_input = 1
    n_reservoir = 10
    spectral_radius = 0.4
    sparsity = 0.9
    input_scaling = 0.1
    leak_rate = 0.9
    echo_random_state = 41
    random_state = 6
    device = 'cpu'

    # Initialize ESN
    esn = EchoStateNetwork(
        n_input=n_input,
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        device=device,
        random_state_torch=echo_random_state,
        random_state_numpy=random_state
    )

    # Generate a random series of 1000 steps
    steps = 100
    outputs = esn.generate_series(steps, roll_every=5)

    # Retrieve hidden states
    hidden_states = esn.get_hidden_states()

    # Plot the generated outputs
    plt.figure(figsize=(12, 6))
    plt.plot(outputs, label='ESN Output')
    plt.title(f'Random Series Generated by Echo State Network seed {random_state}')
    plt.xlabel('Time Steps')
    plt.ylabel('Output Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional: Inspect a hidden state
    # For example, the state at step 500
    # step_to_inspect = 500
    # if step_to_inspect < len(hidden_states):
    #     hidden_state = hidden_states[step_to_inspect].flatten()
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(hidden_state, label=f'Hidden State at Step {step_to_inspect}')
    #     plt.title(f'Reservoir Hidden State at Step {step_to_inspect}')
    #     plt.xlabel('Neuron Index')
    #     plt.ylabel('Activation')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
