import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention import MultiHeadAttention
from .embeddings import Embeddings 
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = LayerNorm(config)
        self.norm2 = LayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, attention_mask=None):

        # attention_mask를 float로 변환하고 매우 작은 값으로 마스킹
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=x.dtype)  # x와 같은 dtype으로 변환
            attention_mask = (1.0 - attention_mask) * -10000.0  # 마스킹된 위치에 매우 작은 값

        # print(f"encoder x = {x.shape}")
        # # 셀프 어텐션 (Pre-LN)
        # residual = x
        # x = self.norm1(x)  # 정규화를 먼저 수행
        # x = self.attention(x, attention_mask)
        # x = residual + self.dropout(x)  # 잔차 연결

        # # 피드포워드 (Pre-LN)
        # residual = x
        # x = self.norm2(x)  # 정규화를 먼저 수행
        # x = self.feed_forward(x)
        # x = residual + self.dropout(x)  # 잔차 연결
        # return x

        # Pre-LN 구조
        residual = x
        x = self.norm1(x)
        # attention_mask가 있으면 float 타입으로 변환
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=x.dtype)
        x = self.attention(x, attention_mask)
        x = residual + self.dropout(x)

        # 피드포워드
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        # 임베딩 처리
        x = self.embeddings(input_ids)
        
        # 인코더 레이어 순차 처리
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        return x