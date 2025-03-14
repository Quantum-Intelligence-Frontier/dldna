import torch
import torch.nn as nn
import torch.nn.functional as F

# v10-5 - Cross-attention with hierarchical feature fusion added
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

        # Projections and normalization for hierarchical feature processing
        self.level_projections = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(3)
        ])
        self.level_weights = nn.Parameter(torch.ones(3) / 3)
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(3)
        ])

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

        # Hierarchical feature processing
        def process_features(features):
            levels = []
            for i, (proj, norm) in enumerate(zip(self.level_projections, self.level_norms)):
                level_feat = norm(proj(features))
                levels.append(level_feat)
            return levels

        image_levels = process_features(image_features)
        text_levels = process_features(text_features)
        level_weights = F.softmax(self.level_weights, dim=0)

        # Attention calculation per level
        image_attended_levels = []
        text_attended_levels = []

        for i in range(3):
            # Pre-LN
            img_norm = self.attn_norm(image_levels[i])
            txt_norm = self.attn_norm(text_levels[i])

            # Image -> Text attention
            q_img = split_heads(self.to_q(img_norm))
            k_txt = split_heads(self.to_k(txt_norm))
            v_txt = split_heads(self.to_v(txt_norm))

            attn_i2t = torch.matmul(q_img, k_txt.transpose(-2, -1)) * self.scale
            attn_i2t = attn_i2t.softmax(dim=-1)
            attn_i2t = self.dropout(attn_i2t)
            img_attended = torch.matmul(attn_i2t, v_txt)

            # Text -> Image attention
            q_txt = split_heads(self.to_q(txt_norm))
            k_img = split_heads(self.to_k(img_norm))
            v_img = split_heads(self.to_v(img_norm))

            attn_t2i = torch.matmul(q_txt, k_img.transpose(-2, -1)) * self.scale
            attn_t2i = attn_t2i.softmax(dim=-1)
            attn_t2i = self.dropout(attn_t2i)
            txt_attended = torch.matmul(attn_t2i, v_img)

            # Combine heads and apply weights
            img_attended = img_attended.transpose(1, 2).reshape(B, N_i, -1)
            txt_attended = txt_attended.transpose(1, 2).reshape(B, N_t, -1)

            image_attended_levels.append(img_attended * level_weights[i])
            text_attended_levels.append(txt_attended * level_weights[i])

        # Fuse level results
        image_attended = sum(image_attended_levels)
        text_attended = sum(text_attended_levels)

        # Output projection and dropout
        image_attended = self.dropout(self.to_out(image_attended))
        text_attended = self.dropout(self.to_out(text_attended))

        # Pre-LN: Normalize before FFN
        image_ff = self.ff_norm(image_attended)
        text_ff = self.ff_norm(text_attended)

        # Gated feedforward processing
        def apply_ff(x):
            gate = self.ff_gate(x)
            value = self.ff_value(x)
            return self.dropout(self.ff_out(gate * value))

        # FFN output and residual connection
        image_out = apply_ff(image_ff) + image_attended
        text_out = apply_ff(text_ff) + text_attended

        return image_out, text_out