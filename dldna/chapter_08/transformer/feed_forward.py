import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import TransformerConfig

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Networks"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # First linear transformation
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Second linear transformation
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Weight initialization
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.zeros_(self.w_1.bias)
        nn.init.zeros_(self.w_2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        # FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        return x