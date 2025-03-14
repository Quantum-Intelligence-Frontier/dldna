import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        self.d_k = config.hidden_size // config.num_attention_heads
        self.h = config.num_attention_heads
        
        print(f"\nInitializing MultiHeadAttention:")
        print(f"d_k (attention_head_size): {self.d_k}")
        print(f"h (num_attention_heads): {self.h}")
        print(f"total hidden size: {config.hidden_size}")
        
        # Linear projections
        self.linear_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size) 
            for _ in range(4)  # Q, K, V, and output
        ])
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = scores.softmax(dim=-1)
        p_attn = self.dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 1) Linear projections
        query = self.linear_layers[0](x)
        key = self.linear_layers[1](x)
        value = self.linear_layers[2](x)
        
        # 2) Split into heads and transpose
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        # 3) Apply attention
        x, attn = self.attention(query, key, value, mask)
        
        # 4) Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.h * self.d_k)
        
        return self.linear_layers[3](x)

