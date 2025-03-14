import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Dict, Any
# from tqdm.notebook import tqdm # Use standard tqdm in distributed setting
import socket
from tqdm import tqdm  # Use standard tqdm
from datetime import timedelta  # Import timedelta

# Assuming these are correctly defined elsewhere in your project.
from dldna.chapter_04.models.base import SimpleNetwork
from dldna.chapter_04.utils.metrics import calculate_metrics, save_results_to_csv
from dldna.chapter_05.visualization.optimization import plot_training_results
from dldna.chapter_05.experiments.basic import get_gpu_memory_info

def find_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup_distributed(rank: int, world_size: int):
    """Sets up the distributed process group.

    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
    """
    try:
        print(f"Process {rank}: Starting setup_distributed")
        port = find_free_port()
        print(f"Process {rank}: Using port {port}")

        # Add timeout
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{port}',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=1)  # Add a timeout
        )
        print(f"Process {rank}: Process group initialized")
    except Exception as e:
        print(f"Process {rank}: Error in setup_distributed: {str(e)}")
        raise


def cleanup_distributed():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def run_distributed_experiment(
    optimizer_class: type,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    epochs: int = 30,
    gpu_id: int = 0,
    world_size: int = 2
) -> Dict[str, Any]:
    """Runs a distributed training experiment.

    Args:
        optimizer_class: The optimizer class to use.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
        config: Configuration dictionary for the optimizer.
        device: The device ('cuda' or 'cpu') to use.  Note: This will be overridden
                by the distributed setup. Each process will get its own device.
        epochs: Number of training epochs.
        gpu_id:  Local GPU ID for this process (0 or 1 typically).
        world_size: Total number of processes/GPUs.

    Returns:
        A dictionary containing training results (only from rank 0).
    """

    # Distributed setup
    rank = gpu_id
    print(f"Initializing distributed process rank {rank}")
    setup_distributed(rank, world_size)


    # Model and optimizer initialization
    print(f"Rank {rank}: Creating model and optimizer")
    model = SimpleNetwork(act_func=nn.GELU()).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = optimizer_class(model.parameters(), **config)
    criterion = nn.CrossEntropyLoss()
    print(f"Rank {rank}: Model and optimizer created")


    # Print initial memory info (only on rank 0)
    if device.type == 'cuda' and rank == 0:
        memory_info = get_gpu_memory_info(gpu_id)
        print(f"\n{'='*50}")
        print(f"Rank {rank}: Optimizer: {optimizer_class.__name__}")
        print(f"Rank {rank}: Initial CUDA Memory Status (GPU {gpu_id}):")
        print(f"Rank {rank}: Allocated: {memory_info['allocated']:.1f}MB")
        print(f"Rank {rank}: Reserved: {memory_info['reserved']:.1f}MB")
        print(f"Rank {rank}: Model Size: {sum(p.numel() for p in model.parameters())/1000:.1f}K parameters")
        print(f"{'='*50}\n")

    results = {
        'epochs': [],
        'train_loss': [], 'test_loss': [],
        'train_accuracy': [], 'test_accuracy': [],
        'train_gradient_norm': [], 'test_gradient_norm': [],
        'memory_allocated': [], 'memory_reserved': []
    }

    for epoch in range(epochs): # Removed tqdm wrapper here.  Use it inside train/eval.
        print(f"Rank {rank}: Starting epoch {epoch+1}/{epochs}")
        # Training loop
        model.train()
        train_metrics = train_epoch_distributed(model, train_loader, optimizer, criterion, device)

        # Evaluation loop
        model.eval()
        test_metrics = evaluate_model_distributed(model, test_loader, criterion, device)

        if rank == 0:  # Only the master process saves results
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
            results['test_gradient_norm'].append(0.0)  # No gradients during eval

    # Final memory status output (only on rank 0)
    if device.type == 'cuda' and rank == 0:
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

    cleanup_distributed()
    if rank == 0:
        save_results_to_csv(results, optimizer_class.__name__, 'distributed')
    return results if rank == 0 else None


def train_epoch_distributed(model, loader, optimizer, criterion, device):
    """Trains the model for a single epoch in a distributed environment."""
    total_loss = torch.zeros(1).to(device)
    correct = torch.zeros(1).to(device)
    total = torch.zeros(1).to(device)
    total_grad_norm = torch.zeros(1).to(device)
    num_batches = len(loader)

    # Only rank 0 displays the progress bar
    rank = dist.get_rank()
    print(f"Process {rank}: Starting training epoch")

    if rank == 0:
        loader = tqdm(loader, desc="Training", leave=False)  # Use standard tqdm

    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"Process {rank}: Processing batch {batch_idx}/{num_batches}")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Calculate gradient norm
        batch_grad_norm = calculate_grad_norm(model.module)  # Use .module for DDP
        total_grad_norm += batch_grad_norm

        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum()
        total += targets.size(0)
    
        # Only rank 0 updates the progress bar
        if rank == 0:
            loader.set_postfix({
                'loss': f'{loss.item():.4f}',
                'batch': f'{batch_idx}/{num_batches}',
                'gpu': f'{rank}'  # Add GPU ID to the progress bar
            })

    print(f"Process {rank}: Finished training, starting all_reduce")

    # Aggregate results from all processes
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_grad_norm, op=dist.ReduceOp.SUM)

    print(f"Process {rank}: Completed all_reduce")

    results = {
        'loss': (total_loss / num_batches).item(),
        'accuracy': (correct / total).item(),
        'grad_norm': (total_grad_norm / num_batches).item()
    }
    print(f"Process {rank}: Final results - Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.4f}")
    return results


def evaluate_model_distributed(model, loader, criterion, device):
    """Evaluates the model in a distributed environment."""
    with torch.no_grad():
        total_loss = torch.zeros(1).to(device)
        correct = torch.zeros(1).to(device)
        total = torch.zeros(1).to(device)
        num_batches = len(loader)

        # Only rank 0 displays the progress bar
        rank = dist.get_rank()
        if rank == 0:
            loader = tqdm(loader, desc="Evaluating", leave=False) # Use standard tqdm


        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum()
            total += targets.size(0)
            
            # Only rank 0 updates the progress bar
            if rank == 0:
                loader.set_postfix({'loss': loss.item()})

        # Aggregate results from all processes
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        return {
            'loss': (total_loss / num_batches).item(),
            'accuracy': (correct / total).item(),
            'grad_norm': 0.0  # No gradients during evaluation
        }

def calculate_grad_norm(model):
    """Calculates the overall gradient norm of the model.

    Args:
        model: PyTorch model (can be wrapped in DistributedDataParallel)

    Returns:
      The calculated gradient norm, as a Python float.  Moves result to the
      same device as the model's parameters to avoid device mismatches during
      all_reduce.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return torch.tensor(total_norm ** 0.5, device=next(model.parameters()).device) # Return as tensor on correct device