# examples/copy_task.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from expertai_src.chapter_07.transformer_sola.config import TransformerConfig
from expertai_src.chapter_07.transformer_sola.transformer import Transformer

def create_simple_sequences(seq_len = 20, batch_size: int = 32) -> torch.Tensor:
    """더 단순하고 해석하기 쉬운 시퀀스 생성"""
    # 작은 어휘 사전 크기 사용 (예: 10)
    vocab_size = 10
    # 고정된 시퀀스 길이
    seq_length = seq_len
    
    sequences = torch.randint(1, vocab_size, (batch_size, seq_length))
    return sequences

def create_copy_data(config, batch_size):
    seq_length = config.max_position_embeddings - 1  # 시작 토큰을 위해 1을 뺌
    input_seq = torch.randint(1, config.vocab_size, (batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones_like(input_seq, dtype=torch.float32)
    return input_seq, attention_mask

def train_copy_task(
    config: TransformerConfig,
    num_epochs: int = 10,
    batch_size: int = 64,
    steps_per_epoch: int = 100
) -> None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"batch_size={batch_size}")
    model = Transformer(config).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    for epoch in range(num_epochs):
        total_loss = 0
        for step in range(steps_per_epoch):
            input_seq, attention_mask = create_copy_data(config, batch_size)
            input_seq, attention_mask = input_seq.to(device), attention_mask.to(device)

            outputs = model(input_seq, attention_mask=attention_mask)
            
            loss = criterion(
                outputs.view(-1, config.vocab_size),
                input_seq.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return model
        

def train_copy_task_pytorch(
    config: TransformerConfig,
    num_epochs: int = 10,
    batch_size: int = 64,
    steps_per_epoch: int = 100
) -> None:
    """PyTorch 공식 Transformer를 사용한 복사 태스크 학습"""
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # PyTorch Transformer 초기화
    model = nn.Transformer(
        d_model=config.hidden_size,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_hidden_layers,
        num_decoder_layers=config.num_hidden_layers,
        dim_feedforward=config.intermediate_size,
        dropout=config.hidden_dropout_prob,
        batch_first=True
    ).to(device)
    
    # 임베딩 레이어 추가
    embedding = nn.Embedding(config.vocab_size, config.hidden_size).to(device)
    
    # 출력 레이어 추가
    output_layer = nn.Linear(config.hidden_size, config.vocab_size).to(device)
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(list(model.parameters()) + 
                         list(embedding.parameters()) + 
                         list(output_layer.parameters()), 
                         lr=0.0001)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for step in range(steps_per_epoch):
            # 데이터 생성 및 GPU로 이동
            input_seq = create_copy_data(batch_size).to(device)
            
            # 임베딩
            src = embedding(input_seq)
            tgt = embedding(input_seq)
            
            # Transformer는 target sequence에서 마지막 토큰 제외
            tgt_mask = model.generate_square_subsequent_mask(input_seq.size(1)).to(device)
            
            # 모델 출력
            output = model(src, tgt, tgt_mask=tgt_mask)
            output = output_layer(output)
            
            # 손실 계산
            loss = criterion(
                output.view(-1, config.vocab_size),
                input_seq.view(-1)
            )
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / steps_per_epoch
        print(f"[PyTorch] Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    return model

def simple_test(model: Transformer):
    """학습된 모델로 간단한 테스트 수행"""
    # 테스트용 시퀀스 생성
    test_seq = create_simple_sequences(batch_size=1)  # 1개 시퀀스만 생성
    print("입력 시퀀스:", test_seq[0])
    
    # 모델 추론
    with torch.no_grad():
        output = model(test_seq, test_seq)
        predicted = torch.argmax(output, dim=-1)
        print("출력 시퀀스:", predicted[0])
        
    # 정확도 확인
    correct = (test_seq == predicted).all()
    print("복사 성공:", correct.item())

def visualize_attention(model: Transformer, input_seq: torch.Tensor):
    """어텐션 패턴 시각화"""
    # 첫 번째 인코더 레이어의 첫 번째 헤드 어텐션 가중치 추출
    with torch.no_grad():
        _ = model(input_seq, input_seq)  # forward pass
        attention_weights = model.encoder.layers[0].attention.attention_probs[0, 0]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.detach().numpy(),
        xticklabels=input_seq[0].tolist(),
        yticklabels=input_seq[0].tolist(),
        cmap='viridis'
    )
    plt.title('Attention Pattern')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.show()

def main():
    # 설정
    # config = TransformerConfig()
    # config.vocab_size = 100
    # config.max_position_embeddings = 20
    
    # # 1. 모델 학습
    # print("=== 모델 학습 시작 ===")
    # model = train_copy_task(config)
    
    # # 2. 간단한 테스트
    # print("\n=== 테스트 시작 ===")
    # simple_test(model)
    
    # # 3. 어텐션 패턴 시각화
    # print("\n=== 어텐션 패턴 시각화 ===")
    # test_seq = create_simple_sequences(batch_size=1)
    # visualize_attention(model, test_seq)


    config = TransformerConfig()
    # 기본값 수정
    config.vocab_size = 20           # 작은 어휘 사전
    config.hidden_size = 128          # 작은 히든 차원
    config.num_hidden_layers = 2      # 최소 레이어 수
    config.num_attention_heads = 2     # 최소 헤드 수
    config.intermediate_size = 128     # 작은 FFN 차원
    config.max_position_embeddings = 10 # 짧은 시퀀스 길이

    model = train_copy_task(config, num_epochs=30, batch_size=32, steps_per_epoch=100)

if __name__ == "__main__":
    main()







    