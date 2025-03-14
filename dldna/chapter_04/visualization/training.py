import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Dict, Any  # Import Dict and Any
import torch
from tqdm.notebook import tqdm

def plot_results(results: Dict[str, Any]) -> None:
    """Visualizes the training results (accuracy).

    Args:
        results: A dictionary containing training and testing accuracies.
                 Expected keys: 'train_accuracies', 'test_accuracies'.
    """
    train_acc = results['train_accuracies']  # Corrected key names
    test_acc = results['test_accuracies']    # Corrected key names
    epochs = range(len(train_acc))

    fig = plt.figure(figsize=(5, 3))
    fig.subplots_adjust(wspace=0.2)

    plt.title('Accuracy', fontdict={'size': 10})
    plt.xlabel('Epochs', fontdict={'size': 10})

    sns.lineplot(x=epochs, y=train_acc, color="C0", label='Train')
    sns.lineplot(x=epochs, y=test_acc, color="C1", label='Test')

    plt.legend()
    plt.show()

def create_results_table(results_dict: Dict[str, Dict[str, float]]) -> None:
    """Prints the training results as a Markdown table.

    Args:
        results_dict: A dictionary containing the results.  The structure
            should be:
            {
                'model_name': {
                    'accuracy': float,
                    'loss': float,
                    'time': float
                }
            }
    """
    print("Model | Accuracy (%) | Final Loss | Time (s)")
    print("-- | -- | -- | --")
    for model_name, metrics in results_dict.items():
        print(f"{model_name} | {metrics['accuracy']:.1f} | {metrics['loss']:.2f} | {metrics['time']:.1f}")


def plot_training_results(model_name: str, results: dict) -> None:
    """Plots training results (loss and accuracy) for a model.

    Args:
        model_name: The name of the model.
        results: A dictionary containing training results.
                 Expected keys: 'epochs', 'train_losses', 'test_losses',
                                'train_scores', 'test_scores'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    # Get epoch values from results
    epochs = results['epochs']

    # Loss graph
    ax1.plot(epochs, results['train_losses'], label='Train')
    ax1.plot(epochs, results['test_losses'], label='Validation')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} Loss')
    min_loss = min(min(results['train_losses']), min(results['test_losses']))
    max_loss = max(max(results['train_losses']), max(results['test_losses']))
    ax1.set_ylim(min_loss * 0.95, max_loss * 1.05)
    ax1.legend()

    # Accuracy graph
    ax2.plot(epochs, results['train_scores'], label='Train')
    ax2.plot(epochs, results['test_scores'], label='Validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} Accuracy')
    min_acc = min(min(results['train_scores']), min(results['test_scores']))
    max_acc = max(max(results['train_scores']), max(results['test_scores']))
    ax2.set_ylim(min_acc * 0.95, max_acc * 1.05)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    plt.close()



def calculate_disabled_neuron(model, data_loader, device):
    """Calculates the number and ratio of disabled (all-zero output) neurons.

    Calculates the number and ratio of neurons that are disabled (i.e., have
    all-zero outputs) after passing through each activation function layer,
    across an entire data loader.

    Args:
        model: The PyTorch model.
        data_loader: The DataLoader for the dataset.
        device: The device ('cpu' or 'cuda') to use.

    Returns:
        None (prints the results).
    """
    act_func_name = model.config["act_func"]
    # Create an array to mask the output of each layer, initialized to all True.
    masked_neuron_array = [torch.ones(layer.weight.shape[0], device=device, dtype=torch.bool)
                           for layer in model.layers[:-1] if "Linear" in str(layer.__class__)]  # Exclude the last layer
    print(f"\nNumber of layers to compare = {len(masked_neuron_array)}")

    model.eval()

    with torch.no_grad():
        for imgs, _ in tqdm(data_loader):  # Iterate through all data.
            idx = 0
            x = imgs.view(imgs.size(0), -1).to(device)  # Reshape input to (batch, 784)

            for layer in model.layers[:-1]:  # Exclude the last layer (output layer)
                x = layer(x)  # Output becomes the input of the next layer.
                if act_func_name in str(layer.__class__):  # Check after activation function layers.
                    all_zero_in_one_batch = (x == 0).all(dim=0)  # True only if all data points in a batch are 0 for a specific index.
                    masked_neuron_array[idx] = torch.logical_and(masked_neuron_array[idx], all_zero_in_one_batch)  # Logical AND with masked array.
                    idx += 1

    num_disabled = [mask.sum().item() for mask in masked_neuron_array]
    print(f"Number of disabled neurons ({model.config['act_func']}) : {num_disabled}")

    # Calculate the percentage per layer.
    for total_neurons, disabled_count in zip(masked_neuron_array, num_disabled):
        print(f"Ratio of disabled neurons = {100 * (disabled_count / total_neurons.shape[0]):>0.1f}%")