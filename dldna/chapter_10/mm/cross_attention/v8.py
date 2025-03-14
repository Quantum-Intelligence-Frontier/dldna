import torch
import torch.nn as nn
import torch.nn.functional as F


# v8 - Independent multi-head
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)

        # Projections for multi-head attention
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Output projection
        self.to_out = nn.Linear(dim, dim)

        # Add output normalization
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, image_features, text_features):
        B, N_i, _ = image_features.shape
        _, N_t, _ = text_features.shape
        H = self.num_heads

        # Input normalization
        image_norm = self.norm(image_features)
        text_norm = self.norm(text_features)

        def split_heads(x):
            return x.reshape(B, -1, H, self.head_dim).transpose(1, 2)

        # Image -> Text direction attention
        q_img = split_heads(self.to_q(image_norm))
        k_txt = split_heads(self.to_k(text_norm))
        v_txt = split_heads(self.to_v(text_norm))

        attn_i2t = torch.matmul(q_img, k_txt.transpose(-2, -1)) * self.scale
        attn_i2t = attn_i2t.softmax(dim=-1)
        image_attended = torch.matmul(attn_i2t, v_txt)

        # Text -> Image direction attention
        q_txt = split_heads(self.to_q(text_norm))
        k_img = split_heads(self.to_k(image_norm))
        v_img = split_heads(self.to_v(image_norm))

        attn_t2i = torch.matmul(q_txt, k_img.transpose(-2, -1)) * self.scale
        attn_t2i = attn_t2i.softmax(dim=-1)
        text_attended = torch.matmul(attn_t2i, v_img)

        # Combine heads and output projection
        image_attended = image_attended.transpose(1, 2).reshape(B, N_i, -1)
        text_attended = text_attended.transpose(1, 2).reshape(B, N_t, -1)

        image_out = self.out_norm(self.to_out(image_attended))
        text_out = self.out_norm(self.to_out(text_attended))

        return image_out, text_out