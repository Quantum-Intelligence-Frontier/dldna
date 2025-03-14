# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .attention import MultiHeadAttention
from .embeddings import Embeddings 
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

# class TransformerDecoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # 셀프 어텐션 레이어
#         self.self_attention = MultiHeadAttention(config)
#         # 교차 어텐션 레이어
#         self.cross_attention = MultiHeadAttention(config)
#         # 피드포워드 네트워크
#         self.feed_forward = FeedForward(config)
        
#         # 레이어 정규화
#         self.norm1 = LayerNorm(config)
#         self.norm2 = LayerNorm(config)
#         self.norm3 = LayerNorm(config)
        
#         # 드롭아웃
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
#         # 셀프 어텐션 (마스크드)
#         self_attention_output = self.self_attention(
#             x, x, x, # 같은 x를 Q, K, V로 사용
#             attention_mask=self_attention_mask
#         )
#         x = self.norm1(x + self.dropout(self_attention_output))

#         # 교차 어텐션
#         cross_attention_output = self.cross_attention(
#             x,              # Q: 디코더의 현재 상태
#             encoder_output, # K: 인코더의 출력
#             encoder_output, # V: 인코더의 출력
#             attention_mask=cross_attention_mask
#         )
#         x = self.norm2(x + self.dropout(cross_attention_output))

#         # 피드포워드
#         ff_output = self.feed_forward(x)
#         x = self.norm3(x + self.dropout(ff_output))

#         return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
        # Pre-LN을 위한 레이어 정규화
        self.norm1 = LayerNorm(config)
        self.norm2 = LayerNorm(config)
        self.norm3 = LayerNorm(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Pre-LN: 정규화를 먼저 수행
        residual = x
        x = self.norm1(x)
        x = self.self_attention(
            x, x, x,
            attention_mask=self_attention_mask
        )
        x = residual + self.dropout(x)

        # 교차 어텐션
        residual = x
        x = self.norm2(x)
        x = self.cross_attention(
            x,
            encoder_output,
            encoder_output,
            attention_mask=cross_attention_mask
        )
        x = residual + self.dropout(x)

        # 피드포워드
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        # N개의 디코더 레이어 스택
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        self.final_norm = LayerNorm(config)

    def forward(self, input_ids, encoder_outputs, 
               self_attention_mask=None, cross_attention_mask=None):
        # 임베딩 처리
        x = self.embeddings(input_ids)
        
        # 디코더 레이어 순차 처리
        for layer in self.layers:
            x = layer(
                x,
                encoder_outputs,
                self_attention_mask,
                cross_attention_mask
            )
            
        return self.final_norm(x)

