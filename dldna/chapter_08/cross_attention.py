import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model=512, num_attention_heads=8, dropout_prob=0.1):
        super().__init__()
        self.d_model = d_model  # 512 (based on the original paper)
        self.num_attention_heads = num_attention_heads  # 8 (based on the original paper)
        self.attention_head_size = int(d_model / num_attention_heads)  # 512/8 = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 8*64 = 512

        # Linear transformations for Q, K, V
        self.query = nn.Linear(d_model, self.all_head_size)
        self.key = nn.Linear(d_model, self.all_head_size)
        self.value = nn.Linear(d_model, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # Input shape: (batch_size, seq_length, d_model)
        batch_size, seq_length, _ = x.size()
        
        # Reshape: (batch_size, seq_length, num_heads, head_size)
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        
        # Final shape: (batch_size, num_heads, seq_length, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, decoder_hidden_states, encoder_outputs, attention_mask=None):
        # decoder_hidden_states: Current state of the decoder (Q)
        # encoder_outputs: Output of the encoder (K, V)
        
        # Linear transformation of Q, K, V
        query_layer = self.query(decoder_hidden_states)  # Q: Decoder state
        key_layer = self.key(encoder_outputs)           # K: Encoder output
        value_layer = self.value(encoder_outputs)       # V: Encoder output

        # Transform Q, K, V into multi-head format
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Calculate attention scores: Q * K^T
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply mask (if provided)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Calculate attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Calculate final attention output
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Restore to original shape
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), -1, self.d_model)

        return context_layer