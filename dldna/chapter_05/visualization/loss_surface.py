import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Any  
import seaborn as sns
from pyhessian import hessian
from matplotlib.tri import Triangulation
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.tri import Triangulation

import seaborn as sns
import torch.nn.functional as F

sns.set_style("darkgrid")


def calculate_accuracy(outputs, labels):
    """Calculates the accuracy of the model's predictions."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def linear_interpolation(model1, model2, data_loader, loss_func, device, num_points=50):
    """Calculates the loss along a linear path between two models."""
    model1 = model1.to(device)
    model2 = model2.to(device)

    alphas = np.linspace(-0.5, 1.5, num_points)  # Extrapolate
    losses = []
    accuracies = []

    for alpha in alphas:
        interpolated_model = copy.deepcopy(model1)
        for p1, p2, p_interp in zip(model1.parameters(),
                                     model2.parameters(),
                                     interpolated_model.parameters()):
            p_interp.data = alpha * p2.data + (1 - alpha) * p1.data

        interpolated_model.to(device)
        interpolated_model.eval()  # Set to evaluation mode

        with torch.no_grad():
            loss = 0
            acc = 0
            total = 0
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = interpolated_model(inputs)
                loss += loss_func(outputs, labels).item() * inputs.size(0)
                acc += calculate_accuracy(outputs, labels) * inputs.size(0)
                total += inputs.size(0)

            loss /= total
            acc /= total
            losses.append(loss)
            accuracies.append(acc)

    return alphas, losses, accuracies


def visualize_linear_interpolation(alphas, losses, accuracies, title, size=(8, 6)):
    """Visualizes the loss and accuracy along the interpolation path."""
    plt.figure(figsize=size)

    # Loss curve
    plt.subplot(2, 1, 1)
    plt.plot(alphas, losses, color='C0', linewidth=2.5)
    plt.fill_between(alphas, losses, color='C0', alpha=0.2)
    plt.plot(alphas[0], losses[0], 'go', markersize=8)  # Start point (green)
    plt.plot(alphas[-1], losses[-1], 'ro', markersize=8)  # End point (red)

    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Accuracy curve
    plt.subplot(2, 1, 2)
    plt.plot(alphas, accuracies, color='C1', linewidth=2.5)
    plt.fill_between(alphas, accuracies, color='C1', alpha=0.2)
    plt.plot(alphas[0], accuracies[0], 'go', markersize=8)  # Start point
    plt.plot(alphas[-1], accuracies[-1], 'ro', markersize=8)  # End point

    plt.xlabel('Interpolation Coefficient (α)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# PCA, t-SNE
def get_weights_as_vector(model):
    """Converts all weights of the model into a 1D vector.

    Args:
        model: The PyTorch model.

    Returns:
        A 1D NumPy array containing all the model weights.
    """
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)

def analyze_weight_space(models, labels, method='pca', perplexity=30):
    """Analyzes the weight space of multiple models using PCA or t-SNE.

    Args:
        models: List of models.
        labels: List of labels for each model.
        method: 'pca' or 'tsne'.
        perplexity: Perplexity parameter for t-SNE.

    Returns:
        The dimensionality-reduced embeddings of the model weights.
    """
    # Extract weight vectors
    weight_vectors = np.array([get_weights_as_vector(model) for model in models])

    if method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity)

    # Perform dimensionality reduction
    embedded = reducer.fit_transform(weight_vectors)

    return embedded

def visualize_weight_space(embedded_weights, labels, method='PCA', size=(6, 4)):
    """Visualizes the dimensionality-reduced weight space.

    Args:
        embedded_weights:  The dimensionality-reduced weight embeddings.
        labels: Labels for each of the models.
        method:  The dimensionality reduction method used ('PCA' or 'TSNE').
        size: Figure size (width, height).
    """
    plt.figure(figsize=size)

    # Plot scatter plot
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        plt.scatter(embedded_weights[mask, 0],
                   embedded_weights[mask, 1],
                   c=[color],  # Wrap color in a list
                   label=label,
                   alpha=0.7)

    plt.title(f'{method} Visualization of Model Weight Space', fontsize=14, pad=15)
    plt.xlabel(f'{method} Component 1', fontsize=12)
    plt.ylabel(f'{method} Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Place the legend outside the plot on the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust margins so the legend doesn't get cut off
    plt.tight_layout()
    plt.show()


# pyHessian

def get_params(model_orig,  model_perb, direction, alpha):
    """Perturbs the parameters of a model along a given direction.

    Args:
        model_orig: The original model.
        model_perb: The model to perturb (a deep copy of the original).
        direction: The direction vector to perturb along.
        alpha: The scaling factor for the perturbation.

    Returns:
      The perturbed model.  Modifies `model_perb` in-place.
    """
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def hessian_eigenvectors(model, loss_func, data_loader, top_n=2, is_cuda=False):
    """Calculates the top n Hessian eigenvectors using PyHessian.

    Args:
        model: The PyTorch model.
        loss_func: The loss function.
        data_loader: DataLoader for the dataset.
        top_n: Number of top eigenvectors to calculate.
        is_cuda: Whether to use CUDA.

    Returns:
      Tuple: (top_eigenvalues, top_eigenvectors)
    """
    (inputs, labels) = next(iter(data_loader))
    model.eval()

    # Instantiate the Hessian computation module.
    hessian_compute = hessian(model, loss_func, data=(inputs,labels), cuda=is_cuda)
    # Calculate the top n eigenvectors.
    top_eigenvalues, top_eignevectors = hessian_compute.eigenvalues(top_n=top_n)

    return top_eigenvalues, top_eignevectors


def xy_perturb_loss(model, top_eigenvectors, data_loader, loss_func, device,
                    lambda1 = np.linspace(-1.0, 1.0, 40).astype(np.float32),
                    lambda2 = np.linspace(-1.0, 1.0, 40).astype(np.float32)):
    """Calculates the loss surface by perturbing the model parameters.

    Perturbs the model parameters along the given eigenvectors using the
    scalars lambda1 and lambda2.

    Args:
        model: The PyTorch model.
        top_eigenvectors: The top eigenvectors of the Hessian.
        data_loader: DataLoader for the dataset.
        loss_func: The loss function.
        device: The device ('cuda' or 'cpu') to use.
        lambda1: Array of scalar values for the first eigenvector.
        lambda2: Array of scalar values for the second eigenvector.

    Returns:
        Tuple: (x, y, z) coordinates of the loss surface.
            - x: Lambda1 values.
            - y: Lambda2 values.
            - z: Loss values.
    """

    # lambda1, lambda2 are scalars used to perturb model parameters along eigenvectors.
    model = model.to(device)
    model_p1 = copy.deepcopy(model).to(device)  # Model for perturbation measurement 1
    model_p2 = copy.deepcopy(model).to(device)  # Model for perturbation measurement 2

    model.eval()
    model_p1.eval()
    model_p2.eval()

    loss_list = []

    inputs, labels = next(iter(data_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    # Calculate all perturbations within the range of lambda. Grid calculation for x, y arrays.
    for l1 in lambda1:
        for l2 in lambda2:
            model_p1 = get_params(model, model_p1, top_eigenvectors[0], l1)
            model_p2 = get_params(model_p1, model_p2, top_eigenvectors[1], l2)
            loss_list.append((l1, l2, loss_func(model_p2(inputs), labels).item()))

    loss_list = np.array(loss_list)

    x, y, z = loss_list[:,0], loss_list[:,1], loss_list[:,2]

    return (x, y, z)


def visualize_loss_surface(data, act_name, color, size=6, alpha=0.7, levels=20, plot_3d=False):
    """Visualizes the loss surface.

    Args:
        data: List of (x, y, z) tuples representing the loss surface.
        act_name: Name of the activation function.
        color: Color to use for the plot.
        size: Size of the figure.
        alpha: Transparency of the surface plot.
        levels: Number of contour levels (for 2D plots).
        plot_3d: Whether to plot in 3D (True) or 2D (False).

    Returns:
        The matplotlib axes object(s).
    """

    cols = len(data)  # Get the number of columns for graph output
    axes = []

    if plot_3d == True:
        fig_size = (size*cols, size)

        fig = plt.figure(figsize=fig_size)

        for i, (x,y,z) in enumerate(data):
            ax = fig.add_subplot(1,cols,i+1, projection="3d")

            ax.plot_trisurf(x, y, z, color=color, alpha=alpha)
            ax.tricontourf(x, y, z, levels=levels, zdir='z', offset=0, cmap=cm.coolwarm)
            ax.set_zlim([0.0, z.max()])

            # Common title and labels
            ax.set_title(f'Loss Landscape ({act_name}) \ntop_eigenvectors=[{i*2}, {i*2+1}]',fontdict = {'fontsize' : 10})
            ax.set_xlabel('lambda 1',fontdict = {'fontsize' : 10})
            ax.set_ylabel('lambda 2',fontdict = {'fontsize' : 10})

            ax.set_zlabel('Loss',fontdict = {'fontsize' : 10})
            ax.view_init(elev=25, azim=90)

            axes.append(ax)

    else :
        fig_size = (size*cols, size-1)

        fig = plt.figure(figsize=fig_size)
        for i, (x,y,z) in enumerate(data):
            ax =  fig.add_subplot(1,cols,i+1)
            triang = Triangulation(x, y)
            t = ax.tricontourf(triang, z, levels=levels, cmap=cm.coolwarm)

             # Common title and labels
            ax.set_title(f'Loss Contour ({act_name}) \ntop_eigenvectors=[{i*2}, {i*2+1}]',fontdict = {'fontsize' : 10})
            ax.set_xlabel('lambda 1',fontdict = {'fontsize' : 10})
            ax.set_ylabel('lambda 2',fontdict = {'fontsize' : 10})

            plt.colorbar(t)

            axes.append(ax)

    # plt.show()  # Don't show here, return the axes instead
    return axes

def persistence_diagram_analysis(x, y, z, act_name, size=(6, 4)):
    """
    Calculates and visualizes the persistence diagram based on the given loss surface (x, y, z data).

    Args:
        x: x-coordinates on the plane (1D array).
        y: y-coordinates on the plane (1D array).
        z: Loss values at the corresponding coordinates (1D array).
        act_name: Name of the activation function (for graph title).
        size: Plot size (width, height).

    Returns:
        persistence diagram (list)
    """
    # grid_size = int(np.sqrt(len(z)))
    # if grid_size * grid_size != len(z):
    #     print("The length of the input data cannot be reconstructed into a square grid.")
    #     return None
    # # Reconstruct into a 2D array
    # Z = np.reshape(z, (grid_size, grid_size))

    grid_size = int(np.sqrt(len(z)))
    if grid_size * grid_size != len(z):
        print("The length of the input data cannot be reconstructed into a square grid.")
        return None

    # Normalize loss values
    Z = np.reshape(z, (grid_size, grid_size))
    Z = (Z - Z.min()) / (Z.max() - Z.min())  # Normalize to 0-1 range


    # Calculate persistence diagram based on CubicalComplex using Gudhi
    try:
        from gudhi import CubicalComplex
    except ImportError:
        print("The Gudhi library is not installed. Please install it and try again.")
        return None

    cubical_complex = CubicalComplex(top_dimensional_cells=Z)
    cubical_complex.persistence()
    diag = cubical_complex.persistence()

    # Remove points with infinite values and use only finite values for plotting
    finite_points = [pt[1] for pt in diag if pt[1][1] != float('inf')]
    if len(finite_points) == 0:
        print("No finite persistence values.")
        return diag
    finite_points = np.array(finite_points)

    plt.figure(figsize=size)
    plt.scatter(finite_points[:, 0], finite_points[:, 1], c='C1', edgecolor='k', label="Persistence Points")
    # Plot y=x diagonal (Birth=Death)
    max_val = np.max(Z)
    plt.plot([0, max_val], [0, max_val], 'k--', label="Diagonal")
    plt.xlabel("Birth", fontsize=12)
    plt.ylabel("Death", fontsize=12)
    plt.title(f"Persistence Diagram ({act_name})", fontsize=14, pad=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return diag


def multiscale_loss_analysis(x, y, z, act_name, size=(10, 6)):
    """
    Performs multiresolution (wavelet) analysis using pywt based on the given loss surface (x, y, z data),
    and visualizes the approximation coefficients and detail coefficients.

    Args:
        x: x-coordinate values on the plane (1D array)
        y: y-coordinate values on the plane (1D array)
        z: Loss values at the corresponding coordinates (1D array)
        act_name: Name of
act_name: Name of the activation function (for graph title)
        size: Plot size (width, height)

    Returns:
        wavelet coefficients (dictionary)
    """
    grid_size = int(np.sqrt(len(z)))
    if grid_size * grid_size != len(z):
        print("The length of the input data cannot be reconstructed into a square grid.")
        return None
    # Reconstruct into a 2D array (e.g., 40x40)
    Z = np.reshape(z, (grid_size, grid_size))

    try:
        import pywt
    except ImportError:
        print("The pywt library is not installed. Please install it and try again.")
        return None

    # Perform 2-level DWT using Haar wavelet
    coeffs = pywt.wavedec2(Z, 'haar', level=2)
    # coeffs[0]: Final approximation coefficients, coeffs[1:], tuples (H, V, D) detail coefficients

    # Plot: The first subplot is the approximation coefficients, the others are the energy of the detail coefficients at each level (H, V, D)
    n_levels = 1 + len(coeffs[1:])
    fig, axes = plt.subplots(1, n_levels, figsize=size)

    # Plot approximation coefficients
    axes[0].imshow(coeffs[0], cmap='viridis', aspect='auto')
    axes[0].set_title("Approx. Coefficients")
    axes[0].axis('off')

    # Plot detail coefficients for each level
    for i, detail in enumerate(coeffs[1:], start=1):
        H, V, D = detail
        # Calculate: Total energy of detail coefficients (sqrt(H^2 + V^2 + D^2))
        detail_energy = np.sqrt(H**2 + V**2 + D**2)
        axes[i].imshow(detail_energy, cmap='viridis', aspect='auto')
        axes[i].set_title(f"Detail Coeff Level {i}")
        axes[i].axis('off')

    plt.suptitle(f"Multiscale Loss Analysis ({act_name})", fontsize=16)
    plt.tight_layout()
    plt.show()

    return coeffs


# def analyze_loss_surface_topology(model, data_loader, loss_func, device,
#                                 lambda_range=(-0.2, 0.2), grid_size=20):
#     """
#     Performs topological analysis of the loss surface.

#     Args:
#         model: The model to analyze.
#         data_loader: DataLoader.
#         loss_func: Loss function.
#         device: Computation device.
#         lambda_range: Tuple specifying the range of lambda1 and lambda2 (default: (-0.2, 0.2)).
#         grid_size: Size of the grid (default: 20).

#     Returns:
#         diag: persistence diagram
#     """
#     # Move the model to the specified device
#     model = model.to(device)

#     # Calculate principal directions using PyHessian
#     top_n = 4
#     is_cuda = device.type == 'cuda'
#     eigenvalues, eigenvectors = hessian_eigenvectors(
#         model,
#         loss_func,
#         data_loader,
#         top_n=top_n,
#         is_cuda=is_cuda
#     )

#     # Generate 2D loss surface data
#     lambda_min, lambda_max = lambda_range
#     lambda1 = np.linspace(lambda_min, lambda_max, grid_size)
#     lambda2 = np.linspace(lambda_min, lambda_max, grid_size)

#     x, y, z = xy_perturb_loss(
#         model=model,
#         top_eigenvectors=eigenvectors[:2],
#         data_loader=data_loader,
#         loss_func=loss_func,
#         lambda1=lambda1,
#         lambda2=lambda2,
#         device=device
#     )

#     # Perform topological analysis
#     diag = persistence_diagram_analysis(x, y, z, "Loss Surface Topology")
#     return diag

def analyze_loss_surface_topology(model, data_loader, loss_func, device,
                                lambda_range=(-0.2, 0.2), grid_size=20):
    """
    Performs topological analysis of the loss surface.

    Args:
        model: The model to analyze.
        data_loader: DataLoader.
        loss_func: Loss function.
        device: Computation device.
        lambda_range: Tuple specifying the range of lambda1 and lambda2 (default: (-0.2, 0.2)).
        grid_size: Size of the grid (default: 20).

    Returns:
        diag: persistence diagram
    """
    # Move the model to the specified device
    model = model.to(device)

    # Calculate principal directions using PyHessian
    top_n = 4
    is_cuda = device.type == 'cuda'
    eigenvalues, eigenvectors = hessian_eigenvectors(
        model,
        loss_func,
        data_loader,
        top_n=top_n,
        is_cuda=is_cuda
    )
    print(f"Eigenvalues: {eigenvalues}")  # Add this line
    print(f"Eigenvectors shape: {eigenvectors[0].shape}")

    # Generate 2D loss surface data
    lambda_min, lambda_max = lambda_range
    lambda1 = np.linspace(lambda_min, lambda_max, grid_size)
    lambda2 = np.linspace(lambda_min, lambda_max, grid_size)

    x, y, z = xy_perturb_loss(
        model=model,
        top_eigenvectors=eigenvectors[:2],
        data_loader=data_loader,
        loss_func=loss_func,
        lambda1=lambda1,
        lambda2=lambda2,
        device=device
    )

    # --- Check z values ---
    print(f"z min: {np.min(z)}, z max: {np.max(z)}, z mean: {np.mean(z)}")
    if np.isnan(z).any() or np.isinf(z).any():
        print("Error: z contains NaN or Inf values!")
        return None  # Or raise an exception

    if np.allclose(z, z.flatten()[0]): # 모든값이 같은지 확인
        print("Error: All z values are the same!")
        return None

    # Perform topological analysis
    diag = persistence_diagram_analysis(x, y, z, "Loss Surface Topology")
    return diag

# Example of multiscale analysis
def analyze_loss_surface_multiscale(model, data_loader, loss_func, device):
    """
    Performs and visualizes multiscale analysis of the loss surface using pywt.

    Args:
        model: The PyTorch model.
        data_loader: The DataLoader.
        loss_func: The loss function.
        device: The device ('cuda' or 'cpu') to use.
    Returns:
      The wavelet coefficients.

    """

    model = model.to(device)
    is_cuda = device.type == 'cuda'
    # Generate 2D loss surface data in the same way
    top_n = 4
    eigenvalues, eigenvectors = hessian_eigenvectors(model, loss_func, data_loader, top_n=top_n, is_cuda=is_cuda)
    lambda1, lambda2 = np.linspace(-0.2, 0.2, 20), np.linspace(-0.2, 0.2, 20)
    x, y, z = xy_perturb_loss(model, eigenvectors[:2], data_loader, loss_func,
                             lambda1=lambda1, lambda2=lambda2, device=device)

    # Perform multiscale analysis
    coeffs = multiscale_loss_analysis(x, y, z, "Loss Surface Multiscale")
    return coeffs