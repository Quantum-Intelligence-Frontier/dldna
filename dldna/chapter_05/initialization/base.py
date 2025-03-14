import numpy as np
import torch
import torch.nn as nn

def safe_initialize(init_func):
    """Decorator for initialization functions"""
    def wrapper(tensor):
        if tensor.dim() >= 2:
            return init_func(tensor)
        else:
            return nn.init.zeros_(tensor)
    return wrapper

@safe_initialize
def init_weights_lecun(tensor):
    """LeCun initialization (1998)"""
    if tensor.dim() < 2:
        return
    fan_in = tensor.size(1)
    std = np.sqrt(1.0 / fan_in)
    nn.init.normal_(tensor, mean=0., std=std)

@safe_initialize
def init_weights_xavier(tensor):
    """Xavier initialization (2010)"""
    if tensor.dim() < 2:
        return
    fan_in, fan_out = tensor.size(1), tensor.size(0)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    nn.init.normal_(tensor, mean=0., std=std)

@safe_initialize
def init_weights_kaiming(tensor):
    """Kaiming(He) initialization (2015)"""
    if tensor.dim() < 2:
        return
    fan_in = tensor.size(1)
    std = np.sqrt(2.0 / fan_in)
    nn.init.normal_(tensor, mean=0., std=std)

@safe_initialize
def init_weights_lmomentum(tensor, alpha=0.81): 
    """L-Momentum Initialization (Zhuang et al., 2024)"""
    if tensor.dim() < 2:
        return
    fan_in = tensor.size(1)
    std_uniform = np.sqrt(6.0 / fan_in)  # For uniform distribution
    nn.init.uniform_(tensor, -std_uniform, std_uniform)

    with torch.no_grad():
        var = (tensor ** 2).mean().item() #  var = tensor.var().item()  # Unbiased estimator. 논문과 일치시키려면 주석처리
        std_lmomentum = np.sqrt(alpha / var)
        tensor.mul_(std_lmomentum)


@safe_initialize
def init_weights_scaled_orthogonal(tensor):
    """Scaled Orthogonal Initialization"""
    if tensor.dim() < 2:
        return

    n = tensor.size(1)
    nn.init.orthogonal_(tensor)
    scale = np.sqrt(2.0 / n)
    with torch.no_grad():
        tensor.mul_(scale)



init_methods = {
    # Historical/educational importance
    'lecun': init_weights_lecun,        # Proposed the first systematic initialization in 1998
    'xavier_normal': nn.init.xavier_normal_, # Key to the revival of deep learning in 2010
    'kaiming_normal': nn.init.kaiming_normal_, # Standard in the ReLU era, 2015

    # Modern standards
    'orthogonal': nn.init.orthogonal_,  # Important in RNN/LSTM
    'scaled_orthogonal': init_weights_scaled_orthogonal, # Optimization of deep neural networks
    'l-momentum' : init_weights_lmomentum, # L-Momentum Initialization.

}


def init_model(model, method='kaiming_normal'):
    """Initialize model weights"""
    if method not in init_methods:
        raise ValueError(f"Unknown initialization method: {method}")

    init_func = init_methods[method]

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:  # Initialize only 2D or higher-dimensional weights
            init_func(param)
        elif 'bias' in name:  # Initialize biases to 0
            nn.init.zeros_(param)