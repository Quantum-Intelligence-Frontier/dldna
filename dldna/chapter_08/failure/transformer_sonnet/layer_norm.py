import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import TransformerConfig

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))  # 스케일 파라미터
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))  # 이동 파라미터
        self.eps = config.layer_norm_eps  # 수치 안정성을 위한 작은 값

    def forward(self, x):
        # 평균과 표준편차 계산 (마지막 차원에 대해)
        mean = x.mean(-1, keepdim=True)
        std = (x - mean).pow(2).mean(-1, keepdim=True).sqrt()
        # 정규화 후 감마와 베타를 적용
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
