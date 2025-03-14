import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List

def plot_optimization_path(results: Dict[str, List[float]], title: str = "Optimization Path"):
    """Visualizes the optimization path.

    Args:
        results: A dictionary where keys are optimizer names and values are lists
                 of (x, y) coordinates representing the optimization path.
        title: The title of the plot.
    """
    plt.figure(figsize=(10, 6))

    for optimizer_name, path in results.items():
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], '.-', label=optimizer_name)

    plt.title(title)
    plt.xlabel('Epoch')  # Consider changing this to something more meaningful, like "Parameter 1"
    plt.ylabel('Loss')  # Or "Parameter 2" if plotting a 2D parameter space.
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_results(results: Dict[str, Dict],
                         metrics: List[str] = None,
                         mode: str = 'all',  # 'train', 'test', 'all'
                         title: str = "Training Results") -> None:
    """Visualizes and compares training results of multiple optimizers.

    Args:
        results: A dictionary of results for each optimizer.  The structure
            should be:
            {
                'optimizer_name': {
                    'train_loss': [ ... ],
                    'test_loss': [ ... ],
                    'train_accuracy': [ ... ],
                    ...
                }
            }
        metrics: A list of metrics to visualize ['loss', 'accuracy', 'gradient_norm'].
        mode:  'train', 'test', or 'all' to select which data to plot.
        title: The title of the plot.
    """
    if metrics is None:
        metrics = ['loss', 'accuracy']

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]  # Make axes iterable even if there's only one plot

    metric_keys = {
        'loss': ('train_loss', 'test_loss'),
        'accuracy': ('train_accuracy', 'test_accuracy'),
        'gradient_norm': ('train_gradient_norm', 'test_gradient_norm'),
        'memory': ('memory_allocated', 'memory_reserved')  # Added memory metrics
    }

    line_styles = {
        'train': '-',
        'test': '--'
    }


    metric_labels = {
        'loss': 'Loss',
        'accuracy': 'Accuracy',
        'gradient_norm': 'Gradient Norm',
        'memory': 'Memory Usage (MB)'  # Added memory unit
    }

    for i, metric in enumerate(metrics):
        ax = axes[i]
        train_key, test_key = metric_keys[metric]

        for optimizer_name, result in results.items():
            epochs = range(1, len(result[train_key]) + 1)

            if mode in ['train', 'all']:
                ax.plot(epochs, result[train_key],
                       label=f'{optimizer_name} (train)',
                       linestyle=line_styles['train'])

            if mode in ['test', 'all']:
                ax.plot(epochs, result[test_key],
                       label=f'{optimizer_name} (test)',
                       linestyle=line_styles['test'])

        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f'{metric_labels[metric]} Curves ({mode})')
        ax.legend()
        ax.grid(True)

    plt.suptitle(f'{title} - {mode.capitalize()} Results')
    plt.tight_layout()
    plt.show()