import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        self.d_k = config.hidden_size // config.num_attention_heads
        self.h = config.num_attention_heads
        
        # Linear projections
        self.linear_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size) 
            for _ in range(4)  # Q, K, V, and output
        ])
        self.dropout = nn.Dropout(p=config.dropout_prob)
        self.attention_weights = None  # attention 가중치 저장용
        
    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = scores.softmax(dim=-1)
        self.attention_weights = p_attn.detach()  # attention 가중치 저장
        p_attn = self.dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and split into h heads
        query, key, value = [
            layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]
        
        # Apply attention
        x, attn = self.attention(query, key, value, mask)
        
        # Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        return self.linear_layers[-1](x)