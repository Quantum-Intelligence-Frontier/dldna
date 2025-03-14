import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, image_features, text_features):
        # Image -> Text attention
        attn_i2t = torch.matmul(image_features, text_features.transpose(-2, -1)) * self.scale
        attn_i2t = attn_i2t.softmax(dim=-1)
        image_attended = torch.matmul(attn_i2t, text_features)

        # Text -> Image attention
        attn_t2i = torch.matmul(text_features, image_features.transpose(-2, -1)) * self.scale
        attn_t2i = attn_t2i.softmax(dim=-1)
        text_attended = torch.matmul(attn_t2i, image_features)

        return image_attended, text_attended