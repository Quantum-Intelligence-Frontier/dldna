import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import numpy as np

# Import relative path, assuming the structure is as before.
from ..utils.metrics import save_model
from ..models.base import SimpleNetwork

def train_loop(model, data_loader, loss_fn, optimizer, device, batch_size):
    """
    Performs the model training loop for a single epoch.
    
    Args:
        model (torch.nn.Module): The model to be trained.
        data_loader (DataLoader): The training data loader.
        loss_fn: The loss function.
        optimizer: The optimizer for updating model parameters.
        device: The device to use for computation.
        batch_size (int): The batch size.
        
    Returns:
        avg_loss (float): The average loss per epoch.
        accuracy (float): The accuracy per epoch.
    """
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for batch_count, (input_data, label_data) in enumerate(data_loader):
        input_data, label_data = input_data.to(device), label_data.to(device)
        preds = model(input_data)
        loss = loss_fn(preds, label_data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label_data.size(0)
        correct += (preds.argmax(dim=-1) == label_data).sum().item()
        count += label_data.size(0)

        if batch_count % 50 == 0:
            current = batch_count * batch_size + input_data.size(0)
            # print(f"Batch {batch_count}: Loss={loss.item():.4f} [{current}/{len(data_loader.dataset)}]")
    
    avg_loss = total_loss / count
    accuracy = correct / count
    return avg_loss, accuracy

def eval_loop(model, data_loader, loss_fn, device):
    """
    Performs the model evaluation loop for a single epoch.
    
    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (DataLoader): The evaluation data loader.
        loss_fn: The loss function.
        device: The device to use for computation.
        
    Returns:
        avg_loss (float): The average loss per epoch.
        accuracy (float): The accuracy per epoch.
    """
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for input_data, label_data in data_loader:
            input_data, label_data = input_data.to(device), label_data.to(device)
            preds = model(input_data)
            loss = loss_fn(preds, label_data)
            total_loss += loss.item() * label_data.size(0)
            correct += (preds.argmax(dim=-1) == label_data).sum().item()
            count += label_data.size(0)
    avg_loss = total_loss / count
    accuracy = correct / count
    # print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
    return avg_loss, accuracy

# def train_model(model, train_loader, test_loader, device, optimizer=None, 
#                 epochs=15, batch_size=128, save_dir="./tmp/models", retrain=True, save_epochs=None):
#     if optimizer is None:
#         optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        
#     model_name = model.config["model_name"]
#     abs_path = os.path.join(save_dir, f"{model_name}.pth")

#     if not retrain and os.path.exists(abs_path):
#         print("Model file already exists. Training stopped.")
#         return

#     results = {
#         "epochs": [],
#         "train_losses": [], 
#         "train_accuracies": [],
#         "test_losses": [], 
#         "test_accuracies": []
#     }
    
#     loss_func = nn.CrossEntropyLoss()
#     start_time = time.time()
#     print(f"\nStarting training for {model_name}.")

#     total_batches = len(train_loader)
#     log_interval = total_batches // 5

#     for epoch in tqdm(range(epochs)):
#         running_loss = 0.0
#         running_accuracy = 0.0  # Changed from running_score to running_accuracy
#         samples_count = 0
        
#         model.train()
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
            
#             optimizer.zero_grad()
#             output = model(data)
#             loss = loss_func(output, target)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item() * len(data)
#             running_accuracy += (output.argmax(1) == target).sum().item() # calculate accuracy
#             samples_count += len(data)

#             if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
#                 current_epoch = epoch + (batch_idx + 1) / total_batches
#                 train_loss = running_loss / samples_count
#                 train_accuracy = running_accuracy / samples_count # Changed variable name here
#                 test_loss, test_accuracy = eval_loop(model, test_loader, loss_func, device) # Changed variable name here
                
#                 results["epochs"].append(current_epoch)
#                 for key, value in zip(["train_losses", "train_accuracies", "test_losses", "test_accuracies"], 
#                                     [train_loss, train_accuracy, test_loss, test_accuracy]):  # Changed key names here
#                     results[key].append(value)
        
#         # Save the model at specified epochs
#         if save_epochs and (epoch + 1) in save_epochs:
#             checkpoint_name = f"{model_name}-epoch{epoch+1}.pth"
#             save_model(model=model, path=save_dir, model_file=checkpoint_name)

#     print(f"Execution completed for {model_name}, Execution time = {(time.time() - start_time):>0.1f} secs")
#     # Save the final model
#     save_model(model=model, path=save_dir)
    
#     return results


def train_model(model, train_loader, test_loader, device, optimizer=None,
                epochs=15, batch_size=128, save_dir="./tmp/models", retrain=True, save_epochs=None):
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    model_name = model.config["model_name"]
    abs_path = os.path.join(save_dir, f"{model_name}.pth")

    if not retrain and os.path.exists(abs_path):
        print("Model file already exists. Training stopped.")
        return

    results = {
        "epochs": [],
        "train_losses": [],
        "train_accuracies": [],
        "test_losses": [],
        "test_accuracies": []
    }

    loss_func = nn.CrossEntropyLoss()
    start_time = time.time()
    print(f"\nStarting training for {model_name}.")

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        running_accuracy = 0.0
        samples_count = 0

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(data)
            running_accuracy += (output.argmax(1) == target).sum().item()
            samples_count += len(data)

        # Calculate average loss and accuracy for the *entire epoch*
        train_loss = running_loss / samples_count
        train_accuracy = running_accuracy / samples_count

        # Evaluate the model *once per epoch*
        test_loss, test_accuracy = eval_loop(model, test_loader, loss_func, device)

        # Record results *once per epoch*
        results["epochs"].append(epoch + 1)  # Append the correct epoch number
        results["train_losses"].append(train_loss)
        results["train_accuracies"].append(train_accuracy)
        results["test_losses"].append(test_loss)
        results["test_accuracies"].append(test_accuracy)


        # Save the model at specified epochs
        if save_epochs and (epoch + 1) in save_epochs:
            checkpoint_name = f"{model_name}-epoch{epoch+1}.pth"
            save_model(model=model, path=save_dir, model_file=checkpoint_name)

    print(f"Execution completed for {model_name}, Execution time = {(time.time() - start_time):>0.1f} secs")
    # Save the final model
    save_model(model=model, path=save_dir)

    return results


def train_all_models(act_functions, train_loader, test_loader, device, 
                    selected_acts=None, epochs=15, batch_size=128, 
                    save_dir="./tmp/models", save_epochs=None):
    """
    Trains models for selected activation functions and saves models at specified epochs.
    
    Args:
        act_functions: Dictionary of activation functions.
        selected_acts: List of activation functions to train. If None, all functions are trained.
        save_epochs: List of epochs to save the model. If None, only the final model is saved.
    """
    results_dict = {}
    
    for act_name, act_class in act_functions.items():
        if selected_acts and act_name not in selected_acts:
            continue
            
        print(f"\nStarting training for: {act_name}")
        act_func = act_class()
        model = SimpleNetwork(act_func=act_func).to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.8)
        
        start_time = time.time()
        results = train_model(model, train_loader, test_loader, device, 
                            optimizer=optimizer, epochs=epochs, 
                            batch_size=batch_size, save_dir=save_dir,
                            save_epochs=save_epochs)
        
        train_time = time.time() - start_time
        results_dict[f"SimpleNetwork-{act_name}"] = {
            'accuracy': results['test_accuracies'][-1] * 100,  # Changed key name here
            'loss': results['test_losses'][-1],
            'time': train_time
        }

        del model  # Release memory
        
    return results_dict


def train_model_with_metrics(model, train_loader, test_loader, device, optimizer=None, 
                           epochs=15, batch_size=128):
    """
    Model training with enhanced metrics.
    Returns:
        - final_loss: Final test loss
        - gradient_norms: Gradient norm history for each layer
        - layer_stats: Statistical information for each layer
        - convergence_speed: Convergence speed metric
    """
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    results = {
        "train_losses": [], "train_accuracies": [],  # Changed key name here
        "test_losses": [], "test_accuracies": [],    # Changed key name here
        "gradient_norms": {},  # Gradient norm per layer
        "layer_stats": {}      # Statistics per layer
    }
    
    # Initialize gradient norms per layer
    for name, _ in model.named_parameters():
        if 'weight' in name:
            results['gradient_norms'][name] = []
    
    loss_func = nn.CrossEntropyLoss()
    best_loss = float('inf')
    convergence_epoch = epochs  # Track convergence epoch
    
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_grad_norms = {name: [] for name in results['gradient_norms'].keys()}
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            
            # Calculate gradient norms
            for name, param in model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    epoch_grad_norms[name].append(grad_norm)
            
            optimizer.step()
        
        # Store average gradient norm per epoch
        for name in results['gradient_norms'].keys():
            avg_norm = np.mean(epoch_grad_norms[name])
            results['gradient_norms'][name].append(avg_norm)
        
        # Collect layer statistics
        layer_stats = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                stats = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'norm': param.data.norm().item()
                }
                layer_stats[name] = stats
        results['layer_stats'][epoch] = layer_stats
        
        # Evaluation
        test_loss, test_accuracy = eval_loop(model, test_loader, loss_func, device) # Changed variable name here
        results['test_losses'].append(test_loss)
        results['test_accuracies'].append(test_accuracy) # Changed key name here
        
        # Check for convergence (if loss doesn't improve by more than 1%, consider it converged)
        if test_loss < best_loss * 0.99:
            best_loss = test_loss
            convergence_epoch = epoch
    
    # Calculate convergence speed (epochs to convergence relative to total epochs)
    convergence_speed = (epochs - convergence_epoch) / epochs
    
    return {
        'final_loss': results['test_losses'][-1],
        'gradient_norms': results['gradient_norms'],
        'layer_stats': results['layer_stats'],
        'convergence_speed': convergence_speed
    }