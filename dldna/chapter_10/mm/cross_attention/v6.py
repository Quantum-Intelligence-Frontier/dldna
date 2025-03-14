import torch
import torch.nn as nn
import torch.nn.functional as F

# v6 - v2 + Q/K/V 변환을 추가한 크로스 어텐션
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)  # 하나의 LayerNorm만 사용
        
        # Q/K/V 변환 추가
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
    def forward(self, image_features, text_features):
        # 입력 정규화
        image_norm = self.norm(image_features)
        text_norm = self.norm(text_features)
        
        # Q/K/V 변환 적용
        q = self.to_q(image_norm)
        k = self.to_k(text_norm)
        v = self.to_v(text_norm)
        
        # 단순 어텐션 계산
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 양방향 특징 결합 (공유 어텐션 행렬 사용)
        image_out = torch.matmul(attn, v)
        text_out = torch.matmul(attn.transpose(-2, -1), self.to_v(image_norm))
        
        return image_out, text_out
