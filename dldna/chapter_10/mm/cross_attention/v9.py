import torch
import torch.nn as nn
import torch.nn.functional as F

# v9 - Dropout before gated FFN, pass through norm at the end -> trainable
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, ff_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        ff_dim = ff_dim or dim * 4

        # Normalization layers for Pre-LN
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

        # Projections for multi-head attention
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # Output projection
        self.to_out = nn.Linear(dim, dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Gated feedforward network
        self.ff_gate = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.ff_value = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.ff_out = nn.Linear(ff_dim, dim)

    def forward(self, image_features, text_features):
        B, N_i, _ = image_features.shape
        _, N_t, _ = text_features.shape
        H = self.num_heads

        def split_heads(x):
            return x.reshape(B, -1, H, self.head_dim).transpose(1, 2)

        # Pre-LN: Normalize before attention
        image_norm = self.attn_norm(image_features)
        text_norm = self.attn_norm(text_features)

        # Image -> Text direction attention
        q_img = split_heads(self.to_q(image_norm))
        k_txt = split_heads(self.to_k(text_norm))
        v_txt = split_heads(self.to_v(text_norm))

        attn_i2t = torch.matmul(q_img, k_txt.transpose(-2, -1)) * self.scale
        attn_i2t = attn_i2t.softmax(dim=-1)
        attn_i2t = self.dropout(attn_i2t)  # Apply dropout to attention weights
        image_attended = torch.matmul(attn_i2t, v_txt)

        # Text -> Image direction attention
        q_txt = split_heads(self.to_q(text_norm))
        k_img = split_heads(self.to_k(image_norm))
        v_img = split_heads(self.to_v(image_norm))

        attn_t2i = torch.matmul(q_txt, k_img.transpose(-2, -1)) * self.scale
        attn_t2i = attn_t2i.softmax(dim=-1)
        attn_t2i = self.dropout(attn_t2i)  # Apply dropout to attention weights
        text_attended = torch.matmul(attn_t2i, v_img)

        # Combine heads and output projection
        image_attended = image_attended.transpose(1, 2).reshape(B, N_i, -1)
        text_attended = text_attended.transpose(1, 2).reshape(B, N_t, -1)

        # Output projection and dropout
        image_attended = self.dropout(self.to_out(image_attended))
        text_attended = self.dropout(self.to_out(text_attended))

        # Residual connection - connecting the original image features makes training impossible.
        # image_attended = image_attended + image_features
        # text_attended = text_attended + text_features

        # Pre-LN: Normalize before FFN
        image_ff = self.ff_norm(image_attended)
        text_ff = self.ff_norm(text_attended)

        # Gated feedforward processing
        def apply_ff(x):
            gate = self.ff_gate(x)
            value = self.ff_value(x)
            return self.dropout(self.ff_out(gate * value))

        # FFN output and residual connection - this type of residual connection is possible.
        image_out = apply_ff(image_ff) + image_attended
        text_out = apply_ff(text_ff) + text_attended

        return image_out, text_out