# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Dict, List
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
# from ..models.base import SimpleNetwork
# from ..utils.data import get_data_loaders, get_dataset, get_device

# # 시각화 스타일 설정
# sns.set_style("whitegrid")
# sns.set_context("notebook", font_scale=1.2)
# plt.rcParams['figure.figsize'] = [20, 8]
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['axes.titlesize'] = 16
# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12


# ##############################################
# # 그래디언트 및 네트워크 시각화 관련 함수
# ##############################################

# def get_gradients_weights(model: nn.Module, data_loader: DataLoader):
#     """
#     한 배치의 데이터를 이용해 모델의 출력 계산, 역전파 후 각 레이어의 가중치와 그래디언트를 산출한다.
#     """
#     imgs, labels = next(iter(data_loader))
#     device = get_device()
#     imgs, labels = imgs.to(device), labels.to(device)
    
#     model.zero_grad()
#     y = model(imgs)
#     loss = F.cross_entropy(y, labels)
#     loss.backward()
    
#     grads = {name.replace("weight", "gradients"): params.grad.data.view(-1).cpu().numpy() 
#              for name, params in model.named_parameters() if "weight" in name}
#     weights = {name.replace("weight", "weights"): params.data.view(-1).cpu().numpy() 
#               for name, params in model.named_parameters() if "weight" in name}
    
#     model.zero_grad()
#     return grads, weights

# def visualize_distribution(model: nn.Module, data: Dict[str, np.ndarray], title: str = "gradients", color: str = "C0"):
#     """
#     주어진 모델의 각 레이어별 그래디언트(또는 가중치) 분포를 Seaborn 히스토그램으로 시각화한다.
#     """
#     sns.set_theme(font_scale=0.6)
#     cols = len(data)
#     fig, axes = plt.subplots(1, cols, figsize=(cols*3, 2.5))
    
#     fig.suptitle(f"{title} distribution ({model.config['act_func']})", fontsize=10, y=1.05)
#     fig.subplots_adjust(wspace=0.5)
    
#     for i, key in enumerate(data):
#         ax = axes[i % cols]
#         sns.histplot(data=data[key], bins=20, ax=ax, color=color, kde=True)
#         ax.set_title(key, fontdict={'fontsize': 10})
#         ax.set_xlabel(f"{title} magnitude", fontdict={'fontsize': 10})
#         ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    
#     plt.show()

# def visualize_network_gradients():
#     """
#     모델(예: SimpleNetwork)의 그래디언트와 가중치 분포를 산출하고 시각화한다.
#     """
#     device = get_device()
#     train_loader, _ = get_data_loaders()
    
#     model_relu = SimpleNetwork(act_func=nn.ReLU()).to(device)
#     grads, _ = get_gradients_weights(model_relu, train_loader)
#     visualize_distribution(model_relu, grads, title="gradients")

# def visualize_gradients(model: nn.Module, 
#                         gradients: Dict[str, np.ndarray],
#                         title: str = "Gradient Distribution"):
#     """
#     그래디언트 분포를 Seaborn 히스토그램으로 시각화한다.
#     (기존 gradients.py의 함수)
#     """
#     sns.set_style("whitegrid")
#     fig, axes = plt.subplots(1, len(gradients), figsize=(15, 4))
    
#     for (name, grad), ax in zip(gradients.items(), axes):
#         sns.histplot(data=grad.flatten(), ax=ax, kde=True)
#         ax.set_title(f'Layer: {name}')
#         ax.set_xlabel('Gradient Value')
    
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()

# def analyze_gradient_flow(gradient_history: List[Dict[str, np.ndarray]]):
#     """
#     여러 훈련 스텝에 걸쳐 기록된 그래디언트 히스토리를 바탕으로,
#     각 레이어별 평균 그래디언트의 변화 흐름을 시각화한다.
#     """
#     plt.figure(figsize=(10, 6))
#     for layer_name in gradient_history[0].keys():
#         grads = [epoch[layer_name].mean() for epoch in gradient_history]
#         plt.plot(grads, label=layer_name)
#     plt.xlabel('Training Step')
#     plt.ylabel('Mean Gradient')
#     plt.title('Gradient Flow Analysis')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def get_model_outputs(model: nn.Module, data_loader: DataLoader, device: str):
#     """
#     모델의 각 층을 지나며 출력값을 기록하여 딕셔너리 형태로 반환한다.
#     """
#     model.eval()
#     outputs = {}
#     with torch.no_grad():
#         imgs, _ = next(iter(data_loader))
#         imgs = imgs.to(device)
#         x = imgs.view(imgs.size(0), -1).to(device)
#         for i, layer in enumerate(model.layers[:-1]):
#             x = layer(x)
#             key = f"layer.{i}.{layer.__class__.__name__}"
#             outputs[key] = x.view(-1).cpu().numpy()
#     return outputs

# def calculate_disabled_neuron(model: nn.Module, data_loader: DataLoader, device: str):
#     """
#     각 활성화 함수 층을 지난 후, 해당 배치에서 모든 값이 0인 뉴런(비활성 뉴런)의 개수 및 비율을 계산한다.
#     """
#     act_func_name = model.config["act_func"]
#     masked_neuron_array = [torch.ones(layer.weight.shape[0], device=device, dtype=torch.bool) 
#                            for layer in model.layers[:-1] if "Linear" in str(layer.__class__)]
#     print(f"\n비교 레이어 갯수 = {len(masked_neuron_array)}")
#     model.eval()
#     with torch.no_grad():
#         for imgs, _ in tqdm(data_loader):
#             x = imgs.view(imgs.size(0), -1).to(device)
#             idx = 0
#             for layer in model.layers[:-1]:
#                 x = layer(x)
#                 if act_func_name in str(layer.__class__):
#                     all_zero_in_one_batch = (x == 0).all(dim=0)
#                     masked_neuron_array[idx] = torch.logical_and(masked_neuron_array[idx], all_zero_in_one_batch)
#                     idx += 1
#     num_disabled = [mask.sum().item() for mask in masked_neuron_array]
#     print(f"비활성 뉴런의 갯수 ({model.config['act_func']}) : {num_disabled}")
#     for mask, disabled in zip(masked_neuron_array, num_disabled):
#         print(f"비활성 뉴런의 비율 = {100 * (disabled / mask.shape[0]):>0.1f}%")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from ..models.base import SimpleNetwork  # Assuming SimpleNetwork is defined
from ..utils.data import get_data_loaders, get_dataset, get_device

# Set visualization style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [20, 8]
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


##############################################
# Functions for visualizing gradients and network
##############################################

def get_gradients_weights(model: nn.Module, data_loader: DataLoader):
    """
    Calculates the weights and gradients of the model after computing the output
    and performing backpropagation using a single batch of data.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader providing the input data.

    Returns:
        Tuple: A tuple containing two dictionaries: (gradients, weights).
               - gradients: Dictionary mapping layer names to flattened gradient arrays.
               - weights: Dictionary mapping layer names to flattened weight arrays.
    """
    imgs, labels = next(iter(data_loader))
    device = get_device()
    imgs, labels = imgs.to(device), labels.to(device)

    model.zero_grad()
    y = model(imgs)
    loss = F.cross_entropy(y, labels)
    loss.backward()

    grads = {name.replace("weight", "gradients"): params.grad.data.view(-1).cpu().numpy()
             for name, params in model.named_parameters() if "weight" in name}
    weights = {name.replace("weight", "weights"): params.data.view(-1).cpu().numpy()
              for name, params in model.named_parameters() if "weight" in name}

    model.zero_grad()  # Reset gradients after calculation
    return grads, weights

def visualize_distribution(model: nn.Module, data: Dict[str, np.ndarray], title: str = "gradients", color: str = "C0"):
    """
    Visualizes the distribution of gradients (or weights) for each layer of the
    given model using Seaborn histograms.

    Args:
        model: The PyTorch model.
        data: Dictionary mapping layer names to flattened gradient/weight arrays.
        title: The title for the plot (default: "gradients").
        color: The color to use for the histograms (default: "C0").
    """
    sns.set_theme(font_scale=0.6)
    cols = len(data)
    fig, axes = plt.subplots(1, cols, figsize=(cols*3, 2.5))

    fig.suptitle(f"{title} distribution ({model.config['act_func']})", fontsize=10, y=1.05)
    fig.subplots_adjust(wspace=0.5)

    for i, key in enumerate(data):
        ax = axes[i % cols]
        sns.histplot(data=data[key], bins=20, ax=ax, color=color, kde=True)
        ax.set_title(key, fontdict={'fontsize': 10})
        ax.set_xlabel(f"{title} magnitude", fontdict={'fontsize': 10})
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

    plt.show()

def visualize_network_gradients():
    """
    Calculates and visualizes the gradient and weight distributions of a model
    (e.g., SimpleNetwork).
    """
    device = get_device()
    train_loader, _ = get_data_loaders()

    model_relu = SimpleNetwork(act_func=nn.ReLU()).to(device)
    grads, _ = get_gradients_weights(model_relu, train_loader)
    visualize_distribution(model_relu, grads, title="gradients")

def visualize_gradients(model: nn.Module,
                        gradients: Dict[str, np.ndarray],
                        title: str = "Gradient Distribution"):
    """
    Visualizes gradient distributions using Seaborn histograms.
    (Existing function from gradients.py)

    Args:
        model: The PyTorch model (not directly used, but kept for consistency).
        gradients: Dictionary of layer names and their corresponding gradients.
        title: Title for the plot.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(gradients), figsize=(15, 4))

    for (name, grad), ax in zip(gradients.items(), axes):
        sns.histplot(data=grad.flatten(), ax=ax, kde=True)
        ax.set_title(f'Layer: {name}')
        ax.set_xlabel('Gradient Value')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def analyze_gradient_flow(gradient_history: List[Dict[str, np.ndarray]]):
    """
    Visualizes the change in the mean gradient of each layer over multiple
    training steps, based on the recorded gradient history.

    Args:
        gradient_history: A list of dictionaries, where each dictionary
                          represents a training step and maps layer names to
                          gradient arrays.
    """
    plt.figure(figsize=(10, 6))
    for layer_name in gradient_history[0].keys():
        grads = [epoch[layer_name].mean() for epoch in gradient_history]
        plt.plot(grads, label=layer_name)
    plt.xlabel('Training Step')
    plt.ylabel('Mean Gradient')
    plt.title('Gradient Flow Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_model_outputs(model: nn.Module, data_loader: DataLoader, device: str):
    """
    Records and returns the output values of each layer in the model as a dictionary.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader providing input data.
        device: The device ('cpu' or 'cuda') to perform computations on.

    Returns:
        A dictionary mapping layer names to flattened output arrays.
    """
    model.eval()
    outputs = {}
    with torch.no_grad():
        imgs, _ = next(iter(data_loader))
        imgs = imgs.to(device)
        x = imgs.view(imgs.size(0), -1).to(device)
        for i, layer in enumerate(model.layers[:-1]):
            x = layer(x)
            key = f"layer.{i}.{layer.__class__.__name__}"
            outputs[key] = x.view(-1).cpu().numpy()
    return outputs

def calculate_disabled_neuron(model: nn.Module, data_loader: DataLoader, device: str):
    """
    Calculates the number and ratio of neurons that are disabled (all values are 0)
    in a batch after passing through each activation function layer.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader providing input data.
        device:  The device ('cpu' or 'cuda') to perform computations on.
    """
    act_func_name = model.config["act_func"]
    masked_neuron_array = [torch.ones(layer.weight.shape[0], device=device, dtype=torch.bool)
                           for layer in model.layers[:-1] if "Linear" in str(layer.__class__)]
    print(f"\nNumber of layers to compare = {len(masked_neuron_array)}")
    model.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(data_loader):
            x = imgs.view(imgs.size(0), -1).to(device)
            idx = 0
            for layer in model.layers[:-1]:
                x = layer(x)
                if act_func_name in str(layer.__class__):
                    all_zero_in_one_batch = (x == 0).all(dim=0)
                    masked_neuron_array[idx] = torch.logical_and(masked_neuron_array[idx], all_zero_in_one_batch)
                    idx += 1
    num_disabled = [mask.sum().item() for mask in masked_neuron_array]
    print(f"Number of disabled neurons ({model.config['act_func']}) : {num_disabled}")
    for mask, disabled in zip(masked_neuron_array, num_disabled):
        print(f"Ratio of disabled neurons = {100 * (disabled / mask.shape[0]):>0.1f}%")