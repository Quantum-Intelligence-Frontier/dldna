import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm.notebook import tqdm
from dldna.chapter_05.initialization.base import init_model
from dldna.chapter_04.experiments.model_training import train_model_with_metrics
import numpy as np

def compute_condition_number(weight_matrix):
    """Computes the condition number of a weight matrix."""
    if weight_matrix.dim() < 2:
        return 1.0
    try:
        # SVD decomposition
        U, S, V = torch.linalg.svd(weight_matrix)
        # Condition number = max singular value / min singular value
        return (S[0] / S[-1]).item()
    except:
        return float('inf')

def estimate_effective_rank(weight_matrix):
    """Estimates the effective rank ratio of a weight matrix."""
    if weight_matrix.dim() < 2:
        return 1.0
    try:
        # SVD decomposition
        U, S, V = torch.linalg.svd(weight_matrix)
        # Sum of singular values
        total_energy = torch.sum(S)
        # Number of singular values whose cumulative energy ratio reaches 95%
        cumsum = torch.cumsum(S, dim=0)
        effective_rank = torch.sum(cumsum < 0.95 * total_energy).item()
        # Ratio of effective rank to the total dimension
        return effective_rank / min(weight_matrix.shape)
    except:
        return 0.0

def analyze_initialization(model_class, init_methods, train_loader, test_loader,
                         epochs=3, batch_size=256, device='cuda'):
    """Performs a deep analysis of various initialization methods.

    Args:
        model_class: The class of the model to be analyzed.
        init_methods: A dictionary of initialization methods.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
        epochs: The number of training epochs.
        batch_size: Batch size for training.
        device: The device to perform computations on ('cuda' or 'cpu').

    Returns:
      A dictionary containing analysis results for each initialization method,
      including loss, training time, weight statistics, gradient norms,
      layer-wise statistics, and convergence speed.
    """
    results = {}

    for method_name, method_func in init_methods.items():  # Iterate through items()
        print(f"\nInitialization method: {method_name}")  # Use method_name
        model = model_class().to(device)
        init_model(model, method=method_name) # Pass method name

        # Initial weight distribution analysis
        weight_stats = analyze_weight_distribution(model)

        # Track the training process
        start_time = time.time()
        training_metrics = train_model_with_metrics(
            model, train_loader, test_loader, device,
            epochs=epochs, batch_size=batch_size
        )
        train_time = time.time() - start_time

        results[method_name] = {  # Use method_name
            'loss': training_metrics['final_loss'],
            'time': train_time,
            'weight_stats': weight_stats,
            'gradient_norm': training_metrics['gradient_norms'],
            'layer_wise_stats': training_metrics['layer_stats'],
            'convergence_speed': training_metrics['convergence_speed']
        }

    return results

def analyze_weight_distribution(model):
    """Analyzes the weight distribution for each layer.

    Args:
      model: The PyTorch model.

    Returns:
      A dictionary containing weight statistics (mean, std, spectral norm,
      condition number, and rank ratio) for each layer.

    """
    stats = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Handle 1D tensors
            if param.dim() < 2:
                layer_stats = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std(correction=0).item(),  # Bessel's correction
                    'spectral_norm': param.data.abs().max().item(),
                    'condition_number': 1.0,
                    'rank_ratio': 1.0
                }
            else:
                # --- CRITICAL: Move to CPU *before* converting to NumPy ---
                param_data_cpu = param.data.cpu()
                layer_stats = {
                    'mean': param_data_cpu.mean().item(),
                    'std': param_data_cpu.std().item(),
                    'spectral_norm': torch.linalg.matrix_norm(param_data_cpu, ord=2).item(),
                    'condition_number': compute_condition_number(param_data_cpu),
                    'rank_ratio': estimate_effective_rank(param_data_cpu)
                }
            stats[name] = layer_stats
    return stats


def create_detailed_analysis_table(results):
    """Prints a detailed analysis result as a Markdown table.
    
    Args:
        results: A dictionary containing analysis results, as returned by
        `analyze_initialization`.
    """
    headers = [
        "Initialization Method", "Error Rate (%)", "Convergence Speed", "Average Condition Number",
        "Spectral Norm", "Effective Rank Ratio", "Execution Time (s)"
    ]

    print(" | ".join(headers))
    print("|".join(["-" * len(h) for h in headers]))

    for method, metrics in results.items():
        stats = metrics['weight_stats']
        avg_condition = np.mean([s['condition_number'] for s in stats.values()])
        avg_spectral = np.mean([s['spectral_norm'] for s in stats.values()])
        avg_rank = np.mean([s['rank_ratio'] for s in stats.values()])

        row = [
            f"{method:12}",
            f"{metrics['loss']:.2f}",
            f"{metrics['convergence_speed']:.2f}",
            f"{avg_condition:.2f}",
            f"{avg_spectral:.2f}",
            f"{avg_rank:.2f}",
            f"{metrics['time']:.1f}"
        ]
        print(" | ".join(row))