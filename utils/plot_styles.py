import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def set_thesis_style():
    """Set global matplotlib parameters for thesis plots."""
    plt.style.use('seaborn-v0_8-whitegrid')

    # Font settings to match LaTeX
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11
    })

    # Figure settings
    plt.rcParams.update({
        'figure.figsize': (5.5, 4),  # Width, height in inches
        'figure.dpi': 300,  # High resolution
        'figure.constrained_layout.use': True,  # Better spacing
        'savefig.dpi': 300,  # Save at high resolution
        'savefig.format': 'pdf',  # PDF for LaTeX
        'savefig.bbox': 'tight',  # Tight bounding box
        'savefig.pad_inches': 0.02  # Minimal padding
    })

    # Line and color settings
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(
            'color', ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133',
                      '#56B4E9']),
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6
    })


def save_thesis_figure(fig, filename, width_fraction=1.0):
    """
    Save figure for thesis with proper formatting.

    Parameters:
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Filename without extension (will save as PDF)
    width_fraction : float
        Fraction of text width (0.0-1.0)
    """
    # LaTeX textwidth is typically around 6.5 inches
    textwidth_inches = 5.5
    fig.set_size_inches(textwidth_inches * width_fraction,
                        textwidth_inches * width_fraction * fig.get_figheight() / fig.get_figwidth())

    # Create the figures directory if it doesn't exist
    import os
    if not os.path.exists('figures/experiments'):
        os.makedirs('figures/experiments')

    # Save with different formats
    fig.savefig(f'figures/experiments/{filename}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(f'figures/experiments/{filename}.png', format='png', dpi=300, bbox_inches='tight')


def example_thesis_plot():
    """Create an example plot with thesis formatting."""
    set_thesis_style()

    # Create example data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot data
    ax.plot(x, y1, label=r'$\sin(x)$')
    ax.plot(x, y2, label=r'$\cos(x)$')

    # Add labels and title
    ax.set_xlabel(r'Time $t$ (s)')
    ax.set_ylabel(r'Amplitude $A$ (V)')
    ax.set_title(r'Example Sine and Cosine Waves')

    # Add legend
    ax.legend(loc='best', frameon=True)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set limits
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)

    # Save the figure
    save_thesis_figure(fig, 'example_plot', width_fraction=0.8)

    return fig, ax


if __name__ == '__main__':
    # Demonstrate the style with an example plot
    example_thesis_plot()
    plt.show()