import torch
from torch import nn
from typing import Dict, Any
from tqdm.notebook import tqdm
from dldna.chapter_04.models.base import SimpleNetwork
from dldna.chapter_04.utils.metrics import calculate_metrics
from dldna.chapter_05.visualization.optimization import plot_training_results


def get_gpu_memory_info(gpu_id: int = 0) -> Dict[str, float]:
    """Returns memory information for a specific GPU.

    Args:
        gpu_id: The ID of the GPU.

    Returns:
        A dictionary containing memory information (in MB).
    """
    return {
        'allocated': torch.cuda.memory_allocated(gpu_id) / 1024**2,
        'reserved': torch.cuda.memory_reserved(gpu_id) / 1024**2,
        'peak_allocated': torch.cuda.max_memory_allocated(gpu_id) / 1024**2,
        'peak_reserved': torch.cuda.max_memory_reserved(gpu_id) / 1024**2
    }


def run_basic_experiment(
    optimizer_class: type,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    epochs: int = 30,
    gpu_id: int = 0
) -> Dict[str, Any]:
    """Runs a basic experiment on FashionMNIST.

    Args:
        optimizer_class: The optimizer class to use.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the testing dataset.
        config: Configuration dictionary for the optimizer.
        device: The device ('cuda' or 'cpu') to use.
        epochs: Number of training epochs.
        gpu_id: ID of the GPU to use (if using CUDA).

    Returns:
      A dictionary containing training results, including loss, accuracy,
      gradient norms, and memory usage over epochs.
    """
    model = SimpleNetwork(act_func=nn.GELU()).to(device)
    optimizer = optimizer_class(model.parameters(), **config)
    criterion = nn.CrossEntropyLoss()

    # Print initial memory information
    if device.type == 'cuda':
        memory_info = get_gpu_memory_info(gpu_id)
        print(f"\n{'='*50}")
        print(f"Optimizer: {optimizer_class.__name__}")
        print(f"Initial CUDA Memory Status (GPU {gpu_id}):")
        print(f"Allocated: {memory_info['allocated']:.1f}MB")
        print(f"Reserved: {memory_info['reserved']:.1f}MB")
        print(f"Model Size: {sum(p.numel() for p in model.parameters())/1000:.1f}K parameters")
        print(f"{'='*50}\n")

    # Dictionary for storing results
    results = {
        'epochs': [],
        'train_loss': [], 'test_loss': [],
        'train_accuracy': [], 'test_accuracy': [],
        'train_gradient_norm': [], 'test_gradient_norm': [],
        'memory_allocated': [], 'memory_reserved': []
    }

    for epoch in tqdm(range(epochs)):
        # Training loop
        model.train()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluation loop
        model.eval()
        test_metrics = evaluate_model(model, test_loader, criterion, device)

        # Store memory status
        if device.type == 'cuda':
            memory_info = get_gpu_memory_info(gpu_id)
            results['memory_allocated'].append(memory_info['allocated'])
            results['memory_reserved'].append(memory_info['reserved'])

        # Store results
        results['epochs'].append(epoch)
        results['train_loss'].append(train_metrics['loss'])
        results['test_loss'].append(test_metrics['loss'])
        results['train_accuracy'].append(train_metrics['accuracy'])
        results['test_accuracy'].append(test_metrics['accuracy'])
        results['train_gradient_norm'].append(train_metrics['grad_norm'])
        results['test_gradient_norm'].append(0.0)  # No gradients during evaluation

    # Print final memory status
    if device.type == 'cuda':
        memory_info = get_gpu_memory_info(gpu_id)
        print(f"\n{'='*50}")
        print(f"Final CUDA Memory Status (GPU {gpu_id}):")
        print(f"Peak Allocated: {memory_info['peak_allocated']:.1f}MB")
        print(f"Peak Reserved: {memory_info['peak_reserved']:.1f}MB")
        print(f"Current Allocated: {memory_info['allocated']:.1f}MB")
        print(f"Current Reserved: {memory_info['reserved']:.1f}MB")
        print(f"{'='*50}\n")

        # Reset memory statistics
        torch.cuda.reset_peak_memory_stats(gpu_id)

    return results


def run_convergence_analysis(
    optimizers: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Dict[str, Any]]:
    """Analyzes the convergence of various optimization algorithms.

    Args:
        optimizers: A dictionary where keys are optimizer names and values
            are tuples of (optimizer_class, config_dict).
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the testing dataset.
        device: The device ('cuda' or 'cpu') to use.

    Returns:
        A dictionary mapping optimizer names to their experiment results.
    """
    results = {}

    for name, (optimizer_class, config) in optimizers.items():
        print(f"\nStarting experiment: {name}")
        exp_results = run_basic_experiment(
            optimizer_class, train_loader, test_loader, config, device
        )
        results[name] = exp_results
        plot_training_results(name, exp_results)

    return results

def train_epoch(model, loader, optimizer, criterion, device):
    """Trains the model for a single epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for the training dataset.
        optimizer: The optimizer to use (or None for no optimization).
        criterion: The loss function.
        device: The device ('cuda' or 'cpu') to use.

    Returns:
        A dictionary containing the average loss, accuracy, and gradient norm
        for the epoch.  Returns 0.0 for gradient norm if optimizer is None.
    """
    total_loss = 0
    correct = 0
    total = 0
    total_grad_norm = 0
    num_batches = len(loader)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Use optimizer only when in training mode
        if optimizer is not None:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backpropagation only when in training mode
        if optimizer is not None:
            loss.backward()
            # Calculate the gradient norm for each batch
            batch_grad_norm = calculate_grad_norm(model)
            total_grad_norm += batch_grad_norm
            optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_accuracy = correct / total
    avg_grad_norm = total_grad_norm / num_batches if optimizer is not None else 0.0

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'grad_norm': avg_grad_norm
    }


def evaluate_model(model, loader, criterion, device):
    """Evaluates the model.

    Args:
        model: The PyTorch model.
        loader: DataLoader for the dataset.
        criterion: The loss function.
        device: The device ('cuda' or 'cpu') to use.

    Returns:
        A dictionary containing the average loss and accuracy.
    """
    model.eval()
    with torch.no_grad():
        return train_epoch(model, loader, None, criterion, device)  # Pass optimizer=None

def calculate_grad_norm(model):
    """Calculates the overall gradient norm of the model.

    Args:
        model: The PyTorch model.

    Returns:
        The L2 norm of the gradients.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5