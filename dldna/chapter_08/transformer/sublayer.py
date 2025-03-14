import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

from .layer_norm import LayerNorm

class SublayerConnection(nn.Module):
    """Residual connection and normalization for Pre-LN structure"""
    def __init__(self, config):
        super().__init__()
        self.norm = LayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor, sublayer: Callable) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            sublayer: Sublayer to be applied (attention or feed_forward)
        Returns:
            Processed tensor
        """
        return x + self.dropout(sublayer(self.norm(x)))