import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import TransformerConfig

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.eps = config.layer_norm_eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = (x - mean).pow(2).mean(-1, keepdim=True).sqrt()
        return self.gamma * (x - mean) / (std + self.eps) + self.beta