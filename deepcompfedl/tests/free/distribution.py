"""
This file aims to test and understand the Dirichlet distribution.
Feel free to modify the following parameters.
"""

from deepcompfedl.task import load_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters
dataset = "cifar10"
num_classes = 10
num_partitions = 10
alpha = 100


# Fetch and analyse the distribution among clients

distr = np.zeros((num_partitions, num_classes))

for partition_id in range(num_partitions):
    # First, fetch the local partition for this client
    trainloader, valloader = load_data(partition_id, num_partitions, alpha, dataset)
    
    # Count in the evaluation batches
    for batch in valloader:
        for label in batch["label"]:
            distr[partition_id, label] += 1
            
    # Count in the training batches
    for batch in trainloader:
        for label in batch["label"]:
            distr[partition_id, label] += 1


def plot_colored_bars(matrix):
    """
    Displays a plot with m horizontal bars, where each bar is made up of n colored segments.
    The length of each segment is proportional to the corresponding value in the matrix.

    Parameters:
        matrix (numpy.ndarray): A matrix of dimensions m x n, where each row represents
                                a horizontal bar, and each column represents a segment.
    """
    m, n = matrix.shape  # Dimensions of the matrix

    # Normalize values so that each row sums to 1
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / row_sums  # Make each row proportional

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 3))  # Adjust the figure size based on m

    # Generate colors for each column
    colors = cm.RdYlGn(np.linspace(0, 1, n))  # Use a colormap to get n distinct colors
        
    # Draw the horizontal bars
    for i in range(m):
        start = 0  # Starting position for the current row
        for j in range(n):
            width = normalized_matrix[i, j]  # Proportional width of the segment
            ax.barh(i, width, left=start, color=colors[j])
            start += width  # Update the starting position for the next segment

    # Configure the axes
    ax.set_yticks(np.arange(m))
    ax.set_yticklabels([f"Client {i}: {int(np.sum(matrix[i,:]))} samples" for i in range(m)])
    ax.set_xlim(0, 1)  # Ensure the total width of each row is 1 (normalized)
    ax.set_xlabel("Class distribution")
    ax.set_xticks([])
    ax.set_title(f"Dirichlet distribution with alpha={alpha}")
    ax.invert_yaxis()  # Invert the y-axis so the first row appears at the top
    plt.tight_layout()
    plt.savefig(f"dirichlet_distribution_{alpha}.png", dpi=300, bbox_inches='tight')


# Plot the result
plot_colored_bars(distr)
