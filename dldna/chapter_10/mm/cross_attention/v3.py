import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)  # Use a single LayerNorm

    def forward(self, image_features, text_features):
        # Input normalization
        image_norm = self.norm(image_features)
        text_norm = self.norm(text_features)

        # Simple attention calculation
        attn = torch.matmul(image_norm, text_norm.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Bidirectional feature fusion
        image_attended = torch.matmul(attn, text_norm)
        text_attended = torch.matmul(attn.transpose(-2, -1), image_norm)

        # Add residual connection
        image_out = image_features + image_attended
        text_out = text_features + text_attended

        return image_out, text_out