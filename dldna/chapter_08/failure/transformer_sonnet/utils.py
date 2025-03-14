import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

def create_addition_data(num_samples: int = 1000, max_digits: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """덧셈 문제 데이터셋 생성
    
    Args:
        num_samples: 생성할 샘플의 수
        max_digits: 최대 자릿수
        
    Returns:
        (x1, x2, y): 첫번째 숫자, 두번째 숫자, 합계를 담은 텐서 튜플
    """
    x1 = torch.randint(0, 10**max_digits, (num_samples,))
    x2 = torch.randint(0, 10**max_digits, (num_samples,))
    y = x1 + x2
    return x1, x2, y

def number_to_tokens(number: int, max_digits: int = 3) -> List[int]:
    """숫자를 토큰 시퀀스로 변환
    
    Args:
        number: 변환할 숫자
        max_digits: 최대 자릿수
        
    Returns:
        각 자릿수를 나타내는 정수 리스트
    """
    return [int(d) for d in f'{number:0{max_digits}}']

def create_padding_mask(seq_length: int, valid_length: int) -> torch.Tensor:
    """패딩 마스크 생성
    
    Args:
        seq_length: 시퀀스 전체 길이
        valid_length: 실제 데이터가 있는 길이
        
    Returns:
        패딩 마스크 텐서 (1: 실제 토큰, 0: 패딩 토큰)
    """
    mask = torch.zeros((seq_length, seq_length))
    mask[:valid_length, :valid_length] = 1
    return mask

def create_causal_mask(seq_length: int) -> torch.Tensor:
    """인과관계 마스크 생성
    
    Args:
        seq_length: 시퀀스 길이
        
    Returns:
        인과관계 마스크 텐서 (하삼각 행렬)
    """
    return torch.tril(torch.ones(seq_length, seq_length))

def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """예측 정확도 계산
    
    Args:
        predictions: 모델의 예측값 (batch_size, num_classes)
        targets: 실제 레이블 (batch_size,)
        
    Returns:
        정확도 (0~1 사이의 값)
    """
    return (predictions.argmax(dim=-1) == targets).float().mean()

def sequence_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    pad_token_id: int = 0
) -> torch.Tensor:
    """시퀀스 전체의 정확도 계산
    
    Args:
        predictions: 모델의 예측값 (batch_size, seq_len, num_classes)
        targets: 실제 레이블 (batch_size, seq_len)
        pad_token_id: 패딩 토큰의 ID
        
    Returns:
        시퀀스 정확도 (0~1 사이의 값)
    """
    mask = targets != pad_token_id
    correct = ((predictions.argmax(dim=-1) == targets) & mask).sum()
    total = mask.sum()
    return correct.float() / total.float()