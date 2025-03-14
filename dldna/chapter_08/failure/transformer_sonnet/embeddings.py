import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import TransformerConfig
from .layer_norm import LayerNorm

# class Embeddings(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(
#             config.vocab_size, 
#             config.hidden_size
#         )
#         self.position_embeddings = nn.Embedding(
#             config.max_position_embeddings,
#             config.hidden_size
#         )
#         self.layer_norm = LayerNorm(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, input_ids):
#         # 입력 시퀀스의 위치 인덱스 생성
#         seq_length = input_ids.size(1)
#         position_ids = torch.arange(
#             seq_length, 
#             dtype=torch.long, 
#             device=input_ids.device
#         )
#         position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

#         # 단어 임베딩과 위치 임베딩을 더함
#         word_embeddings = self.word_embeddings(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         embeddings = word_embeddings + position_embeddings

#         # 정규화와 드롭아웃 적용
#         embeddings = self.layer_norm(embeddings)
#         embeddings = self.dropout(embeddings)

#         #로깅
#         # print(f"Word embeddings range: {word_embeddings.min():.3f} to {word_embeddings.max():.3f}")
#         # print(f"Position embeddings range: {position_embeddings.min():.3f} to {position_embeddings.max():.3f}") 
#         return embeddings


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.layer_norm = LayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, 
            dtype=torch.long, 
            device=input_ids.device
        ).unsqueeze(0).expand_as(input_ids)
        
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)
    
    