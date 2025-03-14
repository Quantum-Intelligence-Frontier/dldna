# config.py
from dataclasses import dataclass

@dataclass

class TransformerConfig:
    vocab_size: int = 20
    hidden_size: int = 64  # 이 값을 64로 변경
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 128
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 10
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2