import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        print(f"Decoder layer input shape: {hidden_states.shape}")
        self_attention_output = self.self_attention(hidden_states, attention_mask=attention_mask)
        print(f"Decoder layer self attention output shape: {self_attention_output.shape}")
        
        cross_attention_output = self.cross_attention(self_attention_output, encoder_hidden_states)
        print(f"Decoder layer cross attention output shape: {cross_attention_output.shape}")
        
        layer_output = self.feed_forward(cross_attention_output)
        print(f"Decoder layer output shape: {layer_output.shape}")
        return layer_output

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, attention_mask)
        return hidden_states