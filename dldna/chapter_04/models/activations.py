import numpy as np
import torch.nn as nn
import torch

# Set seed for reproducibility
np.random.seed(7)
torch.manual_seed(7)

# STAF (Sinusoidal Trainable Activation Function)
class STAF(nn.Module):
    """
    Sinusoidal Trainable Activation Function (STAF).

    A trainable activation function based on a sum of sinusoidal components.
    """
    def __init__(self, tau=25):
        super().__init__()
        self.tau = tau
        self.C = nn.Parameter(torch.randn(tau))
        self.Omega = nn.Parameter(torch.randn(tau))
        self.Phi = nn.Parameter(torch.randn(tau))

    def forward(self, x):
        result = torch.zeros_like(x)
        for i in range(self.tau):
            result += self.C[i] * torch.sin(self.Omega[i] * x + self.Phi[i])
        return result

# TeLU (Trainable exponential Linear Unit)
class TeLU(nn.Module):
    """
    Trainable Exponential Linear Unit (TeLU).

    An activation function with a trainable alpha parameter for the negative region.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

# Swish (Custom Implementation)
class Swish(nn.Module):
    """
    Swish activation function (custom implementation).

    Implements the Swish activation:  f(x) = x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

# Activation function dictionary
act_functions = {
    # Classic activation functions
    "Sigmoid": nn.Sigmoid,     # Binary classification output layer
    "Tanh": nn.Tanh,          # RNN/LSTM

    # Modern basic activation functions
    "ReLU": nn.ReLU,          # CNN default
    "GELU": nn.GELU,          # Transformer standard
    "Mish": nn.Mish,          # Performance/stability balance

    # ReLU variants
    "LeakyReLU": nn.LeakyReLU,# Handles negative inputs
    "SiLU": nn.SiLU,          # Efficient sigmoid
    "Hardswish": nn.Hardswish,# Mobile optimized
    "Swish": Swish,           # Custom implementation

    # Adaptive/trainable activation functions
    "PReLU": nn.PReLU,        # Trainable slope
    "RReLU": nn.RReLU,        # Randomized slope
    "TeLU": TeLU,             # Trainable exponential
    "STAF": STAF             # Fourier-based
}