# efficient_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class TransformerConfig:  # Simplified config for educational purposes
    def __init__(self, vocab_size: int = 30522,
                 hidden_size: int = 256,
                 num_hidden_layers: int = 4,
                 num_attention_heads: int = 8,
                 intermediate_size: int = 512,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 layer_norm_eps: float = 1e-12):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps


class LayerNorm(nn.Module):
    """Layer Normalization (Pre-LN variant)."""
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Embeddings(nn.Module):
    """Token and positional embeddings."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights (Xavier/Glorot)
        nn.init.xavier_uniform_(self.token_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embeddings.weight)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class FlashAttention(nn.Module):
    """
    Simplified FlashAttention using PyTorch's built-in scaled_dot_product_attention.
    This avoids manual tiling/recomputation for educational clarity.
    For *true* FlashAttention performance, use xformers or Triton.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Q, K, V transformations
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        #Xavier Initialization
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask (batch_size, 1, 1, seq_len)
                           or (batch_size, 1, seq_len, seq_len).  0 for masked, 1 for unmasked.

        Returns:
            Attention output tensor (batch_size, seq_len, hidden_size).
        """

        # Linear projections
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)


        # Prepare for multi-head attention
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Use PyTorch's built-in scaled_dot_product_attention
        # This handles FlashAttention, memory-efficient attention, etc., if available.
        attn_output = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=attention_mask, dropout_p=self.dropout.p if self.training else 0.0)

        # Concatenate heads and project back to hidden_size
        attn_output = attn_output.transpose(1, 2).contiguous() # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(hidden_states.size(0), hidden_states.size(1), self.hidden_size) # (batch_size, seq_len, hidden_size)

        return attn_output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (FFN)."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Xavier Initialization
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.zeros_(self.dense1.bias)
        nn.init.zeros_(self.dense2.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = F.gelu(x)  # Use GELU activation
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """A single Transformer encoder layer."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = FlashAttention(config)
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN + Residual Connection + FlashAttention
        attention_output = self.attention(self.norm1(hidden_states), attention_mask)
        hidden_states = hidden_states + attention_output

        # Pre-LN + Residual Connection + FFN
        ffn_output = self.ffn(self.norm2(hidden_states))
        hidden_states = hidden_states + ffn_output

        return hidden_states


class TransformerEncoder(nn.Module):
    """The complete Transformer encoder."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states