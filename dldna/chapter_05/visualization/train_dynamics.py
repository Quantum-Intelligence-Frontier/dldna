import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def visualize_training_dynamics(model, optimizer, train_loader, loss_func, num_epochs, device):
    """Visualizes the dynamic characteristics of the model during training.

    Records the following 5 metrics at each epoch and visualizes them in 5 subplots in a single row:
      1. Loss
      2. Gradient Norm (average of per-batch gradient norms)
      3. Parameter Change (L2 norm of the difference from initial parameters)
      4. Weight Norm (sum of L2 norms of the model's current parameters)
      5. Loss Improvement (loss reduction compared to the previous epoch; 0 for the first epoch)

    Args:
        model (torch.nn.Module): The model to train.
        optimizer: The optimization algorithm to update model parameters.
        train_loader: DataLoader for the training data.
        loss_func: Loss function.
        num_epochs (int): Total number of training epochs.
        device: Device to use for computation (e.g., torch.device("cuda") or torch.device("cpu")).

    Returns:
        dict: {
           'loss': [ ... ],
           'grad_norm': [ ... ],
           'param_change': [ ... ],
           'weight_norm': [ ... ],
           'loss_improvement': [ ... ]
        }
        Values of the recorded metrics for each epoch.
    """
    model.to(device)
    model.train()

    # Copy initial parameters (for parameter change measurement)
    initial_params = [param.detach().clone() for param in model.parameters()]

    epoch_losses = []
    epoch_grad_norms = []
    epoch_param_changes = []
    epoch_weight_norms = []
    epoch_loss_improvements = []

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        running_grad_norm = 0.0
        batch_count = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            optimizer.zero_grad()
            loss.backward()

            # Sum of gradient norms for all parameters in the current batch
            batch_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    batch_grad_norm += param.grad.data.norm(2).item()

            running_grad_norm += batch_grad_norm
            running_loss += loss.item()
            batch_count += 1

            optimizer.step()

        avg_loss = running_loss / batch_count
        avg_grad_norm = running_grad_norm / batch_count

        # Change in parameters compared to initial parameters (sum of L2 norms)
        total_param_change = 0.0
        for init_param, current_param in zip(initial_params, model.parameters()):
            total_param_change += (current_param.detach() - init_param).norm(2).item()

        # Total L2 norm of the model parameters
        total_weight_norm = sum(param.detach().norm(2).item() for param in model.parameters())

        # Per-epoch loss improvement (0 for the first epoch, difference from previous avg_loss after that)
        if epoch == 1:
            loss_improvement = 0.0
        else:
            loss_improvement = epoch_losses[-1] - avg_loss

        epoch_losses.append(avg_loss)
        epoch_grad_norms.append(avg_grad_norm)
        epoch_param_changes.append(total_param_change)
        epoch_weight_norms.append(total_weight_norm)
        epoch_loss_improvements.append(loss_improvement)

        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}, "
              f"Param Change: {total_param_change:.4f}, Weight Norm: {total_weight_norm:.4f}, "
              f"Loss Improvement: {loss_improvement:.4f}")

    # Visualize results with 5 subplots (5 in a row)
    epochs_axis = list(range(1, num_epochs + 1))
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    # Subplot setup: Set grid, title, label, and style for each axis
    axs[0].plot(epochs_axis, epoch_losses, marker='o', color='blue', linestyle='-')
    axs[0].set_title("Epoch vs Loss", fontsize=12)
    axs[0].set_xlabel("Epoch", fontsize=10)
    axs[0].set_ylabel("Loss", fontsize=10)
    axs[0].grid(True)

    axs[1].plot(epochs_axis, epoch_grad_norms, marker='o', color='red', linestyle='-')
    axs[1].set_title("Epoch vs Grad Norm", fontsize=12)
    axs[1].set_xlabel("Epoch", fontsize=10)
    axs[1].set_ylabel("Gradient Norm", fontsize=10)
    axs[1].grid(True)

    axs[2].plot(epochs_axis, epoch_param_changes, marker='o', color='green', linestyle='-')
    axs[2].set_title("Epoch vs Param Change", fontsize=12)
    axs[2].set_xlabel("Epoch", fontsize=10)
    axs[2].set_ylabel("Parameter Change", fontsize=10)
    axs[2].grid(True)

    axs[3].plot(epochs_axis, epoch_weight_norms, marker='o', color='purple', linestyle='-')
    axs[3].set_title("Epoch vs Weight Norm", fontsize=12)
    axs[3].set_xlabel("Epoch", fontsize=10)
    axs[3].set_ylabel("Weight Norm", fontsize=10)
    axs[3].grid(True)

    axs[4].plot(epochs_axis, epoch_loss_improvements, marker='o', color='orange', linestyle='-')
    axs[4].set_title("Epoch vs Loss Improvement", fontsize=12)
    axs[4].set_xlabel("Epoch", fontsize=10)
    axs[4].set_ylabel("Loss Improvement", fontsize=10)
    axs[4].grid(True)

    fig.suptitle("Training Dynamics Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    metrics = {
        "loss": epoch_losses,
        "grad_norm": epoch_grad_norms,
        "param_change": epoch_param_changes,
        "weight_norm": epoch_weight_norms,
        "loss_improvement": epoch_loss_improvements,
    }
    return metrics