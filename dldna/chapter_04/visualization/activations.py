import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit
import seaborn as sns
import math

# Set visualization style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
# plt.rcParams['figure.figsize'] = [20, 8]
plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Model activation functions are defined in models/activations.py.
from dldna.chapter_04.models.activations import act_functions

def compute_gradient_flow(activation, x_range=(-5, 5), y_range=(-5, 5), points=100):
    """
    Computes the 3D gradient flow.

    Calculates the output surface of the activation function for two-dimensional
    inputs and the magnitude of the gradient with respect to those inputs.

    Args:
        activation: Activation function (nn.Module or function).
        x_range (tuple): Range for the x-axis (default: (-5, 5)).
        y_range (tuple): Range for the y-axis (default: (-5, 5)).
        points (int): Number of points to use for each axis (default: 100).

    Returns:
        X, Y (ndarray): Meshgrid coordinates.
        Z (ndarray): Activation function output values.
        grad_magnitude (ndarray): Gradient magnitude at each point.
    """
    x = np.linspace(x_range[0], x_range[1], points)
    y = np.linspace(y_range[0], y_range[1], points)
    X, Y = np.meshgrid(x, y)

    # Stack the two dimensions to create a 2D input tensor (first row: X, second row: Y)
    input_tensor = torch.tensor(np.stack([X, Y], axis=0), dtype=torch.float32, requires_grad=True)

    # Construct the surface as the sum of the activation function outputs for the two inputs
    Z = activation(input_tensor[0]) + activation(input_tensor[1])
    Z.sum().backward()

    grad_x = input_tensor.grad[0].numpy()
    grad_y = input_tensor.grad[1].numpy()
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return X, Y, Z.detach().numpy(), grad_magnitude

def visualize_3d_gradient(activation, title):
    """
    Visualizes the 3D output surface and gradient flow (heatmap) of a given activation function.

    Args:
        activation: Activation function (actual object).
        title (str): The title to use, which should be the activation function's name.

    Returns:
        fig: The generated matplotlib Figure object.
    """
    X, Y, Z, grad_magnitude = compute_gradient_flow(activation)

    fig = plt.figure(figsize=(20, 8))

    # Plot the activation function output surface (3D)
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', antialiased=True,
                            alpha=0.8, linewidth=0, edgecolor='none')
    ax1.view_init(elev=30, azim=120)
    ax1.set_title(f'{title} Activation Surface', pad=20, fontsize=16)
    ax1.set_xlabel('Input X', labelpad=10)
    ax1.set_ylabel('Input Y', labelpad=10)
    ax1.set_zlabel('Activation', labelpad=10)
    cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    cbar1.set_label('Activation Value', rotation=270, labelpad=15)

    # Plot the gradient flow heatmap
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(grad_magnitude, extent=[X.min(), X.max(), Y.min(), Y.max()],
                    origin='lower', cmap='magma', aspect='auto')
    ax2.set_title(f'{title} Gradient Magnitude', pad=20, fontsize=16)
    ax2.set_xlabel('Input X')
    ax2.set_ylabel('Input Y')
    cbar2 = fig.colorbar(im, ax=ax2)
    cbar2.set_label('Gradient Magnitude', rotation=270, labelpad=15)

    fig.suptitle(f'Analysis of {title} Activation Function', y=1.05, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig

def visualize_all_activations():
    """
    Iterates through and visualizes all activation functions defined in models/activations.py,
    showing the 3D activation surface and gradient heatmap.
    """
    # act_functions is defined as each activation function's class or constructor.
    for name, act_class in act_functions.items():
        # Instantiate if it's a constructor
        if isinstance(act_class, type):
            activation = act_class()
        else:
            activation = act_class
        fig = visualize_3d_gradient(activation, name)
        plt.show()