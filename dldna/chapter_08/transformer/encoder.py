import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention import MultiHeadAttention
from .embeddings import Embeddings
from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .sublayer import SublayerConnection

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.sublayer = nn.ModuleList([SublayerConnection(config) for _ in range(2)])
    
    def forward(self, x, attention_mask=None):
        # print(f"EncoderLayer input shape: {x.shape}")
        x = self.sublayer[0](x, lambda x: self.attention(x, x, x, attention_mask))
        # print(f"EncoderLayer after attention shape: {x.shape}")
        output = self.sublayer[1](x, self.feed_forward)
        # print(f"EncoderLayer output shape: {output.shape}")
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = LayerNorm(config)
    
    def forward(self, input_ids, attention_mask=None):
        # print(f"Encoder input_ids shape: {input_ids.shape}")
        # print(f"Encoder attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        
        x = self.embeddings(input_ids)
        # print(f"Encoder after embedding shape: {x.shape}")
        
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)
            # print(f"Encoder after layer {i} shape: {x.shape}")
        
        output = self.norm(x)
        # print(f"Encoder final output shape: {output.shape}")
        return output