import torch
import torch.nn as nn
import math
from typing import Optional

from .config import TransformerConfig
from .layer_norm import LayerNorm

class Embeddings(nn.Module):
    """Class to convert input tokens to embedding vectors"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )

        # Layer normalization and dropout
        self.norm = LayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Constants for position encoding
        self.max_position = config.max_position_embeddings
        self.hidden_size = config.hidden_size

        # Apply Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.token_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embeddings.weight)

    def create_position_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Create position indices"""
        seq_length = x.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=x.device
        )
        return position_ids.unsqueeze(0).expand_as(x)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Token embeddings
        token_embeddings = self.token_embeddings(input_ids)

        # Position embeddings
        position_ids = self.create_position_ids(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Sum embeddings
        embeddings = token_embeddings + position_embeddings

        # Apply normalization and dropout
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings