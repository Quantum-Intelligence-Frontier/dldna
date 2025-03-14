import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .config import TransformerConfig

class Transformer(nn.Module):
    """
    Implementation of The Annotated Transformer
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        # Output layer (converts decoder output to vocabulary size)
        self.generator = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """Apply Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask):
        """Encoder processing"""
        return self.encoder(src, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        """Decoder processing"""
        return self.decoder(tgt, memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_mask: Source mask [batch_size, 1, src_len]
            tgt_mask: Target mask [batch_size, 1, tgt_len]
        Returns:
            output: [batch_size, tgt_len, vocab_size]
        """
        # Encoder-decoder processing
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        
        # Generate final output
        output = self.generator(decoder_output)
        return F.log_softmax(output, dim=-1)