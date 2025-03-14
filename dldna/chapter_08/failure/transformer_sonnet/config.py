class TransformerConfig:
    def __init__(self):
        # 기본 설정값
        self.vocab_size = 30000          # 어휘 사전 크기
        self.hidden_size = 768           # 히든 레이어 차원
        self.num_hidden_layers = 12      # 인코더/디코더 레이어 수
        self.num_attention_heads = 12    # 어텐션 헤드 수
        self.intermediate_size = 3072    # FFN 중간 레이어 차원
        self.hidden_dropout_prob = 0.1   # 히든 레이어 드롭아웃
        self.attention_probs_dropout_prob = 0.1  # 어텐션 드롭아웃
        self.max_position_embeddings = 512  # 최대 시퀀스 길이
        self.layer_norm_eps = 1e-5     # 레이어 정규화 입실론