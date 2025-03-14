import torch
import torch.nn as nn
import torch.nn.functional as F


# v7_2 - v2 + added shared multi-head attention
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)  # Keep LayerNorm from v2

        # Projections for multi-head attention
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Output projection
        self.to_out = nn.Linear(dim, dim)

    def forward(self, image_features, text_features):
        B, N_i, _ = image_features.shape
        _, N_t, _ = text_features.shape
        H = self.num_heads

        # Input normalization (kept from v2)
        image_norm = self.norm(image_features)
        text_norm = self.norm(text_features)

        def split_heads(x):
            return x.reshape(B, -1, H, self.head_dim).transpose(1, 2)

        # Q/K/V transformation and head splitting
        q = split_heads(self.to_q(image_norm))
        k = split_heads(self.to_k(text_norm))
        v = split_heads(self.to_v(text_norm))

        # Multi-head attention calculation
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Bidirectional feature fusion
        image_attended = torch.matmul(attn, v)
        image_attended = image_attended.transpose(1, 2).reshape(B, N_i, -1)
        # image_out = self.to_out(image_attended)

        # Text-direction attention (using shared attention matrix)
        text_attended = torch.matmul(attn.transpose(-2, -1), v)  
        text_attended = text_attended.transpose(1, 2).reshape(B, N_t, -1)
        # text_out = self.to_out(text_attended)

        # Apply normalization to the output (otherwise training fails)
        image_out = self.norm(self.to_out(image_attended))
        text_out = self.norm(self.to_out(text_attended))


        return image_out, text_out