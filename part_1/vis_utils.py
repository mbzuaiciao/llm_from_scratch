import os
import numpy as np
import matplotlib.pyplot as plt

# Define the default output directory for saved visualizations.
OUT_DIR = os.path.join(os.path.dirname(__file__), 'out')


def _ensure_out():
    """Ensures that the output directory exists."""
    os.makedirs(OUT_DIR, exist_ok=True)


def save_matrix_heatmap(mat: np.ndarray, title: str, filename: str, xlabel: str = '', ylabel: str = ''):
    """
    Saves a NumPy matrix as a heatmap image.
    - mat: The 2D NumPy array to visualize.
    - title: The title of the plot.
    - filename: The name of the file to save the plot to.
    - xlabel, ylabel: Optional labels for the axes.
    """
    # Ensure the output directory exists.
    _ensure_out()
    # Create a new figure for the plot.
    plt.figure()
    # Display the matrix as an image; `aspect='auto'` adjusts image aspect to fit the axes.
    plt.imshow(mat, aspect='auto')
    # Set the title and axis labels.
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Add a color bar to indicate the scale of the values.
    plt.colorbar()
    # Construct the full save path and save the figure.
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    # Close the figure to free up memory.
    plt.close()
    print(f"Saved: {path}")


def save_attention_heads_grid(weights: np.ndarray, filename: str, title_prefix: str = "Head"):
    """
    Plots attention weights from all heads in a single grid figure.
    Assumes the input is from a single batch item (B=1).
    - weights: A NumPy array of attention weights with shape (1, H, T, T).
    - filename: The name of the file to save the plot to.
    - title_prefix: A prefix for the title of each subplot (e.g., "Head").
    """
    # Ensure the output directory exists.
    _ensure_out()
    # Get the number of heads (H) and sequence length (T) from the weights shape.
    _, H, T, _ = weights.shape
    # Determine the grid layout (e.g., up to 4 columns).
    cols = min(4, H)
    rows = (H + cols - 1) // cols
    # Create a figure with a suitable size for the grid.
    plt.figure(figsize=(3*cols, 3*rows))
    # Loop through each attention head to create its subplot.
    for h in range(H):
        ax = plt.subplot(rows, cols, h+1)
        # Display the attention matrix for the current head.
        ax.imshow(weights[0, h], aspect='auto')
        # Set the title and axis labels for the subplot.
        ax.set_title(f"{title_prefix} {h}")
        ax.set_xlabel('Key pos')
        ax.set_ylabel('Query pos')
    # Adjust subplot params for a tight layout.
    plt.tight_layout()
    # Construct the full save path and save the figure.
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    # Close the figure to free up memory.
    plt.close()
    print(f"Saved: {path}")