import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple, Optional
import torch.nn.functional as F
from dldna.chapter_08.transformer.config import TransformerConfig
from dldna.chapter_08.transformer.transformer import Transformer
from dldna.chapter_08.transformer.masking import create_pad_mask, create_subsequent_mask
import matplotlib.patheffects as PathEffects

# Model save path setup
MODEL_PATH = "saved_models"
MODEL_FILE = "transformer_copy_task.pth"

def save_model(model: Transformer, config: TransformerConfig) -> None:
    """Save the model and config"""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': vars(config)
    }
    torch.save(save_dict, os.path.join(MODEL_PATH, MODEL_FILE))
    print(f"Model saved to {os.path.join(MODEL_PATH, MODEL_FILE)}")

def load_model() -> Tuple[Transformer, TransformerConfig]:
    """Load the saved model and config"""
    model_path = os.path.join(MODEL_PATH, MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")
    
    saved_dict = torch.load(model_path)
    
    config = TransformerConfig()
    for key, value in saved_dict['config'].items():
        setattr(config, key, value)
    
    model = Transformer(config)
    model.load_state_dict(saved_dict['model_state_dict'])
    
    return model, config

def explain_copy_data(seq_length: int = 5) -> None:
    """Explain the format and meaning of the generated copy task data"""
    src, target = create_copy_data(batch_size=1, seq_length=seq_length)
    
    print("\n=== Copy Task Data Explanation ===")
    print(f"Sequence Length: {seq_length}")
    
    print("\n1. Input Sequence:")
    print(f"Original Tensor Shape: {src.shape}")
    print(f"Input Sequence: {src[0].tolist()}")
    
    print("\n2. Target Sequence:")
    print(f"Original Tensor Shape: {target.shape}")
    print(f"Target Sequence: {target[0].tolist()}")
    
    print("\n3. Task Description:")
    print("- Basic task of copying the input sequence as is")
    print("- Tokens at each position are integer values between 1-19")
    print("- Input and output have the same sequence length")
    print(f"- Current Example: {src[0].tolist()} → {target[0].tolist()}")


def create_copy_data(batch_size: int = 32, seq_length: int = 5) -> torch.Tensor:
    """Generate data for the copy task"""
    sequences = torch.randint(1, 20, (batch_size, seq_length))
    return sequences, sequences

class CopyLoss(nn.Module):
    """Loss function for the copy task"""
    def __init__(self):
        super().__init__()
        self.step_counter = 0
    
    def forward(self, outputs: torch.Tensor, target: torch.Tensor, 
                print_details: bool = False) -> Tuple[torch.Tensor, float]:
        batch_size = outputs.size(0)
        self.step_counter += 1
        
        predictions = F.softmax(outputs, dim=-1)
        target_one_hot = F.one_hot(target, num_classes=outputs.size(-1)).float()
        
        loss = -torch.sum(target_one_hot * torch.log(predictions + 1e-10)) / batch_size
        
        with torch.no_grad():
            pred_tokens = predictions.argmax(dim=-1)
            exact_match = (pred_tokens == target).all(dim=1).float()
            match_rate = exact_match.mean().item()
            
        # if print_details and self.step_counter % 5000 == 0:
        #     self._print_details(pred_tokens, target, exact_match, loss, match_rate)
            
        return loss, match_rate

    def _print_details(self, pred_digits, target, exact_match, loss, match_rate):
        """Print detailed information"""
        print(f"\n=== Loss Calculation Details (Step: {self.step_counter}) ===")
        print(f"Predicted Sequences: {pred_digits[:10]}")
        print(f"Actual Sequences: {target[:10]}")
        print(f"Exact Match: {exact_match[:10]}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Accuracy: {match_rate:.4f}")

def train_copy_task(
    config: TransformerConfig,
    num_epochs: int = 50,
    batch_size: int = 64,
    steps_per_epoch: int = 100,
    seq_length: int = 5,
    print_progress: bool = True
) -> Transformer:
    """Train the copy task"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = CopyLoss().to(device)

    print(f"\n=== Start Training ==== \nDevice: {device}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = total_accuracy = 0

        for step in range(steps_per_epoch):
            # Data generation and preprocessing
            src, tgt = create_copy_data(batch_size, seq_length)
            src, tgt = src.to(device), tgt.to(device)
            
            # Prepare decoder input
            decoder_input = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            decoder_input = torch.cat([decoder_input, tgt[:, :-1]], dim=1)
            
            # Create masks
            src_mask = create_pad_mask(src).to(device)
            tgt_mask = create_subsequent_mask(decoder_input.size(1)).to(device)
            
            # Forward and backward passes
            optimizer.zero_grad()
            outputs = model(src, decoder_input, src_mask, tgt_mask)
            loss, accuracy = criterion(outputs, tgt, 
                                    print_details=(step + 1) % steps_per_epoch == 0 
                                    and print_progress)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy

        # Print epoch results
        avg_loss = total_loss / steps_per_epoch
        avg_accuracy = total_accuracy / steps_per_epoch

        if epoch==num_epochs:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    save_model(model, config)
    return model

def test_copy(seq_length: int = 5) -> None:
    """Test with the trained model"""
    try:
        model, config = load_model()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        src, _ = create_copy_data(batch_size=1, seq_length=seq_length)
        src = src.to(device)
        
        with torch.no_grad():
            decoder_input = torch.zeros((1, 1), dtype=torch.long, device=device)
            output_sequence = []
            
            for _ in range(seq_length):
                src_mask = create_pad_mask(src).to(device)
                tgt_mask = create_subsequent_mask(decoder_input.size(1)).to(device)
                
                output = model(src, decoder_input, src_mask, tgt_mask)
                next_token = output[:, -1:].argmax(dim=-1)
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                output_sequence.append(next_token.item())

        print(f"\n=== Copy Test ===")
        print(f"Input: {src[0].cpu().tolist()}")
        print(f"Output: {output_sequence}")
        print(f"Accuracy: {src[0].cpu().tolist() == output_sequence}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Train the model first.")

# def visualize_attention(seq_length: int = 5) -> None:
#     """Visualize the attention pattern"""
#     try:
#         model, _ = load_model()
#         device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         model = model.to(device)
#         model.eval()

#         src, _ = create_copy_data(batch_size=1, seq_length=seq_length)
#         src = src.to(device)
#         src_mask = create_pad_mask(src).to(device)

#         with torch.no_grad():
#             encoder_output = model.encoder(src, src_mask)
#             attention_weights = model.encoder.layers[0].attention.attention(
#                 encoder_output, encoder_output, encoder_output, src_mask
#             )[1]
#             attention_map = attention_weights[0, 0].cpu()

#         plt.figure(figsize=(10, 8))
#         sns.heatmap(attention_map, 
#                    annot=True, 
#                    fmt='.2f',
#                    cmap='YlOrRd',
#                    square=True,
#                    xticklabels=src[0].cpu().tolist(),
#                    yticklabels=src[0].cpu().tolist())
        
#         plt.title('Self-Attention Pattern\nEncoder Layer 0', pad=20)
#         plt.xlabel('Key Tokens')
#         plt.ylabel('Query Tokens')
#         plt.tight_layout()
#         plt.show()

#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         print("Train the model first.")



def visualize_attention(seq_length: int = 5) -> None:
    """Visualize the attention pattern"""
    try:
        model, _ = load_model()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        src, _ = create_copy_data(batch_size=1, seq_length=seq_length)
        src = src.to(device)
        src_mask = create_pad_mask(src).to(device)

        with torch.no_grad():
            encoder_output = model.encoder(src, src_mask)
            # attention_weights의 shape: (batch_size, num_heads, seq_len, seq_len)
            attention_weights = model.encoder.layers[0].attention.attention(
                encoder_output, encoder_output, encoder_output, src_mask
            )[1]
            # 첫 번째 헤드의 어텐션 가중치 사용, shape: (seq_len, seq_len)
            attention_map = attention_weights[0, 0].cpu()

        plt.figure(figsize=(10, 9))  # 조금 더 세로로 긴 그림
        # 'viridis' 컬러맵 사용, annot=True로 숫자 표시, fmt='.2f'로 소수점 둘째 자리까지
        ax = sns.heatmap(attention_map,
                   annot=True,
                   fmt='.2f',
                   cmap='viridis',  # 변경된 컬러맵
                   square=True,
                   linewidths=.5,    # 셀 사이에 라인 추가
                   cbar_kws={"shrink": .82}, # 컬러바 크기 조절
                   xticklabels=src[0].cpu().tolist(),
                   yticklabels=src[0].cpu().tolist())

        # x, y 축 레이블 스타일 변경
        for label in ax.get_xticklabels():
            label.set_fontsize(12) # 폰트 크기
            label.set_fontweight('bold') # 폰트 굵기
            
        for label in ax.get_yticklabels():
            label.set_fontsize(12)
            label.set_fontweight('bold')
            label.set_rotation(0)  # y축 레이블 회전 제거


        # 제목 스타일 변경
        plt.title('Self-Attention Pattern\nEncoder Layer 0',
                  fontsize=16,
                  fontweight='bold',
                  pad=25) # 제목 위쪽 여백
        plt.xlabel('Key Tokens', fontsize=14, labelpad=15)  # x축 레이블
        plt.ylabel('Query Tokens', fontsize=14, labelpad=15) # y축 레이블

        # 대각선 강조 (선택적)
        for i in range(attention_map.shape[0]):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='white', lw=3))
            # 텍스트에 테두리 효과 추가
            # ax.texts는 heatmap 생성 후에 결정되므로, heatmap이 그려진 후에 접근해야 함.
            index = i * attention_map.shape[0] + i  # 올바른 인덱스 계산
            if index < len(ax.texts):  # 텍스트 객체가 존재하는지 확인
                text = ax.texts[index]
                text.set_path_effects([
                    PathEffects.withStroke(linewidth=2, foreground='black')
                ])

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Train the model first.")

def main():
    if not os.path.exists(os.path.join(MODEL_PATH, MODEL_FILE)):
        print("=== Start Model Training ===")
        config = TransformerConfig()
        config.vocab_size = 20
        config.hidden_size = 128
        config.num_hidden_layers = 3
        config.num_attention_heads = 4
        config.intermediate_size = 512
        config.max_position_embeddings = 10
        
        model = train_copy_task(config, seq_length=5)
        
        print("\n=== Copy Test ===")
        test_copy(seq_length=5)
        
        print("\n=== Visualize Attention Pattern ===")
        visualize_attention(seq_length=5)

if __name__ == "__main__":
    main()