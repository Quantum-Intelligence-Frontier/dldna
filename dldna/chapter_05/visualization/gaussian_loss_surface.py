import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from matplotlib.tri import Triangulation
from dldna.chapter_05.visualization.loss_surface import visualize_loss_surface
import matplotlib.pyplot as plt
import numpy as np

def gaussian_func(xy, A, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian function.

    Args:
        xy: Tuple of (x, y) coordinates.
        A: Amplitude.
        x0, y0: Coordinates of the center.
        sigma_x, sigma_y: Standard deviations in x and y directions.
        offset: Offset value.
    """
    x, y = xy
    return -A * np.exp(-(((x-x0)**2/(2*sigma_x**2)) +
                        ((y-y0)**2/(2*sigma_y**2)))) + offset

def get_opt_params(x, y, z):
    """Finds the parameters of the Gaussian function closest to the 3D surface.

    Args:
        x, y: Coordinates on the plane.
        z: Loss values at each coordinate.
    Returns:
        popt: Optimized parameters [A, x0, y0, sigma_x, sigma_y]
        pcov: Covariance matrix of the optimized parameters
        offset: The offset
    """
    offset = z.max()
    popt, pcov = curve_fit(lambda xy, A, x0, y0, sigma_x, sigma_y:
                          gaussian_func(xy, A, x0, y0, sigma_x, sigma_y, offset),
                          (x, y), z)
    print(f"Function parameters = {popt}")
    return popt, pcov, offset

def visualize_gaussian_fit(x, y, z, popt, offset, d_min, d_max, d_num,
                         figsize=(7,7), elev=10, azim=100):
    """Visualizes the original loss surface and the fitted Gaussian function in 3D.

    Args:
        x, y, z: Original loss surface data.
        popt: Gaussian function parameters (A, x0, y0, sigma_x, sigma_y).
        offset: Offset of the Gaussian function.
        d_min, d_max: Range of x and y axes.
        d_num: Number of grid points.
        figsize: Figure size.
        elev, azim: 3D view angles.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Original data scatter plot
    ax.scatter(x, y, z, color='blue', alpha=0.2, label='Original Loss Surface')

    # Gaussian fitting surface
    x_range = np.linspace(d_min, d_max, d_num)
    y_range = np.linspace(d_min, d_max, d_num)
    X, Y = np.meshgrid(x_range, y_range)
    Z = gaussian_func((X, Y), *popt, offset)

    ax.plot_surface(X, Y, Z, color='red', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=elev, azim=azim)
    plt.title('Loss Surface with Gaussian Fit')
    plt.show()



def loss_curve(x, y, A, x0, y0, sigma_x, sigma_y, offset):
    """Gaussian loss function for PyTorch.

    Args:
        x, y: Input coordinates.
        A, x0, y0, sigma_x, sigma_y: Gaussian function parameters.
        offset: Offset value.
    """
    return -A * torch.exp(-(((x-x0)**2/(2*sigma_x**2)) +
                           ((y-y0)**2/(2*sigma_y**2)))) + offset

def train_loss_surface(optim_func, init_weights, num_iter, gaussian_params):
    """Calculates the optimization path on the Gaussian loss surface.

    Args:
        optim_func: Optimizer function generator.
        init_weights: Initial weights.
        num_iter: Number of iterations.
        gaussian_params: Gaussian function parameters (A, x0, y0, sigma_x, sigma_y, offset).

    Returns:
        Numpy array of the optimization path points. Each point has the format [x, y, loss].
    """
    w = nn.Parameter(torch.FloatTensor(init_weights), requires_grad=True)
    optimizer = optim_func([w])

    print(f"\ntrain_loss_surface: {type(optimizer).__name__}")
    points = []

    for i in range(num_iter):
        loss = loss_curve(w[0], w[1], *gaussian_params)
        points.append(torch.cat([w.data.detach(),
                               loss.unsqueeze(dim=0).detach()], dim=0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == 0 or i == num_iter-1:
            print(f"{type(optimizer).__name__}: Iter={i+1} loss={loss:.4f} "
                  f"w=[{w[0]:.4f}, {w[1]:.4f}]")

    return torch.stack(points, dim=0).numpy()

def visualize_optimization_path(x, y, z, popt, offset, optimizers_points,
                              act_name, d_min=-1, d_max=1, d_num=30):
    """Visualizes the optimization path.

    Args:
        x, y, z: Original loss surface data.
        popt: Gaussian function parameters.
        offset: Offset value.
        optimizers_points: Path points of each optimizer.
        act_name: Name of the activation function.
        d_min: Minimum value of the x and y range.
        d_max: Maximum value of the x and y range.
        d_num: Number of points in the grid.

    Returns:
       The matplotlib axes object.
    """
    z_gaussian = gaussian_func((x, y), *popt, offset)
    data = [(x, y, z_gaussian)]

    axes = visualize_loss_surface(data, act_name=act_name, color="C0",
                                size=6, levels=20, alpha=0.7, plot_3d=False)
    ax = axes[0]

    colors = ["C1", "C2", "C3"]
    markers = ["o", "s", "X"]
    labels = ["SGD", "SGD Momentum", "Adam"]

    for points, color, marker, label in zip(optimizers_points, colors,
                                          markers, labels):
        ax.plot(points[:,0], points[:,1], color=color, marker=marker,
               markersize=3, zorder=len(optimizers_points), label=label)

    ax.ticklabel_format(axis='both', style='scientific', scilimits=(0, 0))
    plt.legend()
    plt.show()

    return ax