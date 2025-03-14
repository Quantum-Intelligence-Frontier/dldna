import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import TransformerConfig

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = nn.GELU()  # GELU 활성화 함수 사용
        
        # Xavier 초기화 적용
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.zeros_(self.w_1.bias)
        nn.init.zeros_(self.w_2.bias)

    def forward(self, x):
        x = self.act(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        return x