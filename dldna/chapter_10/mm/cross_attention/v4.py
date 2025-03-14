import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)  # Keep LayerNorm
        self.image_proj = nn.Linear(dim, dim)  # Add image projection
        self.text_proj = nn.Linear(dim, dim)   # Add text projection

    def forward(self, image_features, text_features):
        # Input normalization
        image_norm = self.norm(image_features)
        text_norm = self.norm(text_features)

        # Apply projection
        image_proj = self.image_proj(image_norm)
        text_proj = self.text_proj(text_norm)

        # Simple attention calculation
        attn = torch.matmul(image_proj, text_proj.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Bidirectional feature fusion
        image_out = torch.matmul(attn, text_proj)
        text_out = torch.matmul(attn.transpose(-2, -1), image_proj)

        return image_out, text_out