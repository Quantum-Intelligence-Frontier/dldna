import torch
import torch.nn as nn
from typing import Dict, Any
from tqdm.notebook import tqdm
from dldna.chapter_04.models.vision import CIFAR100CNN as CNN
from dldna.chapter_05.models.transformer import SimpleTransformer
from dldna.chapter_04.utils.metrics import calculate_metrics
from dldna.chapter_05.visualization.optimization import plot_training_results
from dldna.chapter_05.experiments.basic import get_gpu_memory_info
from dldna.chapter_04.utils.metrics import save_results_to_csv

def run_advanced_experiment(
    optimizer_class: type,
    model_type: str,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    epochs: int = 50,
    gpu_id: int = 0
) -> Dict[str, Any]:
    """Advanced experiment for CIFAR-100 and Transformer models"""

    # Model selection
    if model_type == 'cnn':
        model = CNN(num_classes=100).to(device)
    elif model_type == 'transformer':
        model = SimpleTransformer().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = optimizer_class(model.parameters(), **config)
    criterion = nn.CrossEntropyLoss()

    # Initial memory information output
    if device.type == 'cuda':
        memory_info = get_gpu_memory_info(gpu_id)
        print(f"\n{'='*50}")
        print(f"Optimizer: {optimizer_class.__name__}")
        print(f"Initial CUDA Memory Status (GPU {gpu_id}):")
        print(f"Allocated: {memory_info['allocated']:.1f}MB")
        print(f"Reserved: {memory_info['reserved']:.1f}MB")
        print(f"Model Size: {sum(p.numel() for p in model.parameters())/1000:.1f}K parameters")
        print(f"{'='*50}\n")

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

        # Save memory status
        if device.type == 'cuda':
            memory_info = get_gpu_memory_info(gpu_id)
            results['memory_allocated'].append(memory_info['allocated'])
            results['memory_reserved'].append(memory_info['reserved'])

        # Save results
        results['epochs'].append(epoch)
        results['train_loss'].append(train_metrics['loss'])
        results['test_loss'].append(test_metrics['loss'])
        results['train_accuracy'].append(train_metrics['accuracy'])
        results['test_accuracy'].append(test_metrics['accuracy'])
        results['train_gradient_norm'].append(train_metrics['grad_norm'])
        results['test_gradient_norm'].append(0.0)

    # Final memory status output
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

    save_results_to_csv(results, optimizer_class.__name__, model_type)
    return results


def train_epoch(model, loader, optimizer, criterion, device):
    """Single epoch training"""
    total_loss = 0
    correct = 0
    total = 0
    total_grad_norm = 0
    num_batches = len(loader)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Calculate gradient norm
        batch_grad_norm = calculate_gradient_norm(model)
        total_grad_norm += batch_grad_norm

        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_accuracy = correct / total
    avg_grad_norm = total_grad_norm / num_batches

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'grad_norm': avg_grad_norm
    }

def evaluate_model(model, loader, criterion, device):
    """Model evaluation"""
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(loader)

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

    return {
        'loss': total_loss / num_batches,
        'accuracy': correct / total,
        'grad_norm': 0.0  # Gradients are not calculated during evaluation
    }


def calculate_gradient_norm(model):
    """Calculates the total gradient norm of the model"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def calculate_layer_statistics(model):
    """Calculates layer-wise statistics"""
    stats = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'norm': param.data.norm().item()
            }
    return stats