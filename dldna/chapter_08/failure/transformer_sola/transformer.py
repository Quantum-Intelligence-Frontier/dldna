import torch.nn as nn
from .embeddings import Embeddings
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
import torch

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, decoder_input_ids=None, attention_mask=None):
        if decoder_input_ids is None:
            decoder_input_ids = input_ids

        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        
        decoder_embedding_output = self.embeddings(decoder_input_ids)
        decoder_outputs = self.decoder(decoder_embedding_output, encoder_outputs, attention_mask)
        
        lm_logits = self.lm_head(decoder_outputs)
        return lm_logits.to(torch.float32)  # 명시적으로 float32로 변환