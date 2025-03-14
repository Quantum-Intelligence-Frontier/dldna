import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .config import TransformerConfig

# class Transformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.encoder = TransformerEncoder(config)
#         self.decoder = TransformerDecoder(config)
#         self.generator = nn.Linear(config.hidden_size, config.vocab_size)
#         self._init_weights()

#     def _init_weights(self):
#         """가중치 초기화 개선"""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#             elif p.dim() == 1:
#                 nn.init.zeros_(p)
        
#     def _create_masks(self, src, tgt):
#         # 내부적으로 마스크 생성
#         src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
#         tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))
#         return src_mask, tgt_mask
        
#     def forward(self, src, tgt):
#         # 자동으로 마스크 생성 및 처리
#         src_mask, tgt_mask = self._create_masks(src, tgt)
#         encoder_out = self.encoder(src, src_mask)
#         decoder_out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
#         return self.generator(decoder_out)


# transformer.py
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.generator = nn.Linear(config.hidden_size, config.vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt=None):
        """
        수정된 forward 메서드
        Args:
            src: 소스 시퀀스 [batch_size, src_len]
            tgt: 타겟 시퀀스 (옵션) [batch_size, tgt_len]
        """
        # copy task에서는 src를 tgt로 사용
        if tgt is None:
            tgt = src.clone()
        
        # 디코더 입력 준비 (시작 토큰 추가)
        decoder_input = tgt[:, :-1]  # 마지막 토큰 제외
        
        # 마스크 생성
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(1)).to(src.device)
        
        # 인코더-디코더 처리
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(decoder_input, encoder_output, src_mask, tgt_mask)
        
        # 최종 출력 생성
        output = self.generator(decoder_output)
        return F.log_softmax(output, dim=-1)
    
    def _generate_square_subsequent_mask(self, sz):
        """디코더의 subsequent 마스크 생성"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask == 0