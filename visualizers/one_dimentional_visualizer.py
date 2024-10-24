import matplotlib.pyplot as plt
import numpy as np

from generators.poisson_process_generator import PoissonProcessDataset


# Define a helper function to plot the samples
def plot_poisson_samples(dataset, num_samples=5):
    """
    Plots a few samples from the dataset with the same class shown in the same color.

    Args:
        dataset (PoissonProcessDataset): The dataset to sample from.
        num_samples (int): Number of samples to plot from each class.
    """
    plt.figure(figsize=(12, 6))

    # Plot num_samples for each unique intensity (class)
    for i in range(10):
        color = np.random.rand(3, )  # Generate a random color for each class
        subseries, label = dataset[i]
        print(subseries)
        plt.plot(subseries[0], color=color, alpha=0.2, label=f'Intensity {dataset.intensities[label]:.2f}')


    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title(f'Poisson Process Subseries Samples (per Intensity Class)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1, fontsize=8)
    plt.show()


if __name__ == "__main__":
    # Instantiate the dataset
    dataset = PoissonProcessDataset(batch_size=1, series_length=100, subseries_length=50, stride=50)

    # Plot few samples from the dataset
    plot_poisson_samples(dataset)