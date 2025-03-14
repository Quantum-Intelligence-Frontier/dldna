import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states, attention_mask=None):
        print(f"EncoderLayer input shape: {hidden_states.shape}")
        attention_output = self.attention(hidden_states, attention_mask)
        print(f"EncoderLayer attention output shape: {attention_output.shape}")
        layer_output = self.feed_forward(attention_output)
        print(f"EncoderLayer output shape: {layer_output.shape}")
        return layer_output

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states