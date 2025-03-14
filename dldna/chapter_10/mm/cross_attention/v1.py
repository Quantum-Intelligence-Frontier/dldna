import torch
import torch.nn as nn
import torch.nn.functional as F

# Bidirectional feature fusion.  No single LN
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, image_features, text_features):
        # Simple attention calculation
        attn = torch.matmul(image_features, text_features.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Bidirectional feature fusion (using a shared attention matrix)
        image_out = torch.matmul(attn, text_features)
        text_out = torch.matmul(attn.transpose(-2, -1), image_features)

        return image_out, text_out