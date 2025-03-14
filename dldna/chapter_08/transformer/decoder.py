import torch
import torch.nn as nn
import math
from typing import Optional

from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .embeddings import Embeddings
from .layer_norm import LayerNorm

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
        # Layer normalization for Pre-LN
        self.norm1 = LayerNorm(config)
        self.norm2 = LayerNorm(config)
        self.norm3 = LayerNorm(config)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # Pre-LN structure
        m = self.norm1(x)
        x = x + self.dropout(self.self_attn(m, m, m, tgt_mask))
        
        m = self.norm2(x)
        x = x + self.dropout(self.cross_attn(m, memory, memory, src_mask))
        
        m = self.norm3(x)
        x = x + self.dropout(self.feed_forward(m))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)  # Add embedding layer
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = LayerNorm(config)
    
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.embeddings(x)  # Convert input to embeddings
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)