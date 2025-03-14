import torch
import torch.nn as nn
import torch.nn.functional as F

# v5 - v2 + apply only a fixed mixing ratio
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)  # Use a single LayerNorm
        self.mix_ratio = 0.5  # Fixed mixing ratio

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

        # Mix with a fixed ratio (without residual connection)
        image_out = image_norm * (1 - self.mix_ratio) + image_attended * self.mix_ratio
        text_out = text_norm * (1 - self.mix_ratio) + text_attended * self.mix_ratio

        return image_out, text_out