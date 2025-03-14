import torch
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from .data import get_device  # Assuming get_device is in the same package
# from typing import Dict, Any  # Already imported above
import csv
from pathlib import Path
from datetime import datetime
from dldna.chapter_04.models.activations import TeLU, STAF, Swish  # Relative import


def save_model(model, path="./tmp/models", model_file=None):
    """Saves the model and its configuration.

    Args:
        model: The PyTorch model to save.
        path: The directory to save the model in.
        model_file: The filename for the model. If None, uses the model name.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    config = model.config
    config_file = os.path.join(path, f"{config['model_name']}.config")
    with open(config_file, 'w') as f:
        json.dump(config, f)

    if model_file is None:
        model_file = config["model_name"] + ".pth"

    abs_path = os.path.join(path, model_file)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config
    }, abs_path)


def load_model(model_file: str,
              path: str,
              device: Optional[str] = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Loads a model and its configuration.

    Args:
        model_file: The name of the model file (e.g., "SimpleNetwork-ReLU.pth").
        path: The path to the directory containing the model file.
        device: The device to load the model onto (default: None, uses get_device()).

    Returns:
        Tuple: The loaded model and its configuration.

    Raises:
        ValueError: If the activation function or model type is not supported.
    """
    from ..models.base import SimpleNetwork  # Dynamic import
    from ..models.vision import CIFAR100CNN
    from torch import nn

    path = Path(path)
    device = device if device is not None else get_device()

    # Load configuration file
    base_name = "-".join(Path(model_file).stem.split("-")[:2])  # SimpleNetwork-ReLU-epoch9 -> SimpleNetwork-ReLU
    config_file = path / f"{base_name}.config"
    with open(config_file) as f:
        config = json.load(f)



    # Select the appropriate model class based on model type
    if "SimpleNetwork" in config["model_name"]:
        # Activation function setup
        act_name = config.get("act_func", "ReLU")

        # Activation function mapping
        activation_functions = {
            # Classic activation functions
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),

            # Modern basic activation functions
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "Mish": nn.Mish(),

            # ReLU variants
            "LeakyReLU": nn.LeakyReLU(),
            "SiLU": nn.SiLU(),
            "Hardswish": nn.Hardswish(),
            "Swish": Swish(),

            # Adaptive/trainable activation functions
            "PReLU": nn.PReLU(),
            "RReLU": nn.RReLU(),
            "TeLU": TeLU(),
            "STAF": STAF()
        }

        if act_name not in activation_functions:
            raise ValueError(f"Unsupported activation function: {act_name}")

        act_func = activation_functions[act_name]
        model = SimpleNetwork(
            act_func=act_func,
            input_shape=config.get("input_shape", 784),
            num_labels=config.get("num_labels", 10),
            hidden_shape=config.get("hidden_shape", [256, 192, 128, 64])
        )

    elif "CIFAR100CNN" in config["model_name"]:
        model = CIFAR100CNN(num_classes=config.get("num_classes", 100))
    else:
        raise ValueError(f"Unsupported model type: {config['model_name']}")

    # Load the model weights and move it to the specified device
    model = model.to(device)
    checkpoint = torch.load(path / model_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, config



def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Calculates model performance metrics.

    Args:
        outputs: Model output values (N, num_classes).
        targets: True labels (N,).

    Returns:
        Dict[str, float]: Calculated metrics.
    """
    with torch.no_grad():
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(outputs, targets).item()

        # Calculate accuracy
        pred = outputs.argmax(dim=1)
        correct = pred.eq(targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total

        # Calculate gradient norm (optional)
        grad_norm = 0.0
        # Check if gradients exist (they won't during evaluation)
        if outputs.grad is not None:
            grad_norm = outputs.grad.norm().item()


        return {
            'loss': loss,
            'accuracy': accuracy,
            'grad_norm': grad_norm
        }

def calculate_epoch_metrics(model: torch.nn.Module,
                          loader: torch.utils.data.DataLoader,
                          device: torch.device) -> Dict[str, float]:
    """Calculates metrics for an entire epoch.

    Args:
      model: The PyTorch model.
      loader: The DataLoader for the dataset.
      device: The device to use for computation.

    Returns:
        A dictionary containing the average loss and accuracy for the epoch.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            metrics = calculate_metrics(outputs, targets)

            total_loss += metrics['loss'] * targets.size(0)
            total_correct += metrics['accuracy'] * targets.size(0)
            total_samples += targets.size(0)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }




def save_results_to_csv(results: Dict[str, Any], optimizer_name: str, model_type: str, save_dir: str = ".") -> None:
    """Saves experiment results to a CSV file.

    Args:
        results: Dictionary of experiment results.
        optimizer_name: Name of the optimizer used.
        model_type: Type of model ('cnn' or 'transformer').
        save_dir: Directory to save the CSV file.
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create filename (including timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{optimizer_name}_{model_type}_{timestamp}.csv"
    filepath = save_path / filename

    # Save to CSV file
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy',
                        'train_gradient_norm', 'test_gradient_norm', 'memory_allocated', 'memory_reserved'])

        # Write data
        for i in range(len(results['epochs'])):
            writer.writerow([
                results['epochs'][i],
                results['train_loss'][i],
                results['test_loss'][i],
                results['train_accuracy'][i],
                results['test_accuracy'][i],
                results['train_gradient_norm'][i],
                results['test_gradient_norm'][i],
                results['memory_allocated'][i] if results['memory_allocated'] else '',
                results['memory_reserved'][i] if results['memory_reserved'] else ''
            ])

    print(f"Results saved to: {filepath}")




def load_models_by_pattern(model_dir="tmp/models",
                         activation_types=None,
                         epochs=None):
    """Loads only models corresponding to the specified activation functions and epochs.

    Args:
        model_dir: Directory where model files are stored.
        activation_types: List of activation functions to load (e.g., ['ReLU', 'Tanh']).
        epochs: List of epochs to load (e.g., [1,3,5]).  None loads all epochs.

    Returns:
        models: List of loaded models.
        labels: List of labels for each model.
    """
    models = []
    labels = []

    # Get list of files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

    for model_file in sorted(model_files):
        # Check activation function type
        act_type = model_file.split('-')[1]  # SimpleNetwork-ReLU-epoch1.pth -> ReLU
        if activation_types and act_type not in activation_types:
            continue

        # Check epoch
        if 'epoch' in model_file:
            epoch_num = int(model_file.split('epoch')[-1].split('.')[0])
            if epochs and epoch_num not in epochs:
                continue

        # Load model
        model, _ = load_model(model_file=model_file, path=model_dir)
        models.append(model)

        # Create label (e.g., "ReLU-epoch1")
        label = model_file.split('-', 1)[1].rsplit('.', 1)[0]
        labels.append(label)

    return models, labels