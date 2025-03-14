import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple, Optional
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F 


from dldna.chapter_08.transformer.config import TransformerConfig
from dldna.chapter_08.transformer.transformer import Transformer
from dldna.chapter_08.transformer.masking import create_pad_mask, create_subsequent_mask

# Model save path setup
MODEL_PATH = "saved_models"
MODEL_FILE = "transformer_addition_task.pth"

def explain_addition_data(max_digits: int = 3) -> None:
    """Explain the format and meaning of the generated addition data"""
    src, target = create_addition_data(batch_size=1, max_digits=max_digits)
    
    num1 = int(''.join(map(str, src[0, :max_digits].tolist())))
    num2 = int(''.join(map(str, src[0, max_digits+1:].tolist())))
    result = int(''.join(map(str, target[0].tolist())))
    
    print("\n=== Addition Data Explanation ====")
    print(f"Maximum Digits: {max_digits}")
    print("\n1. Input Sequence:")
    print(f"Original Tensor Shape: {src.shape}")
    print(f"First Number: {num1} (Indices {list(src[0, :max_digits].numpy())})")
    print(f"Plus Sign: '+' (Index {src[0, max_digits].item()})")
    print(f"Second Number: {num2} (Indices {list(src[0, max_digits+1:].numpy())})")
    print(f"Full Input: {src[0].tolist()}")
    
    print("\n2. Target Sequence:")
    print(f"Original Tensor Shape: {target.shape}")
    print(f"Actual Sum: {num1 + num2}")
    print(f"Target Sequence: {target[0].tolist()}")


def save_model(model: Transformer, config: TransformerConfig) -> None:
    """Save the model and configuration"""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': vars(config)
    }
    torch.save(save_dict, os.path.join(MODEL_PATH, MODEL_FILE))
    print(f"Model saved to {os.path.join(MODEL_PATH, MODEL_FILE)}")

def load_model() -> Tuple[Transformer, TransformerConfig]:
    """Load the saved model and configuration"""
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


class AdditionLoss(nn.Module):
    """Loss function for the addition task"""
    def __init__(self):
        super().__init__()
        self.step_counter = 0
    
    def forward(self, outputs: torch.Tensor, target: torch.Tensor, print_details: bool = False) -> Tuple[torch.Tensor, float]:

        batch_size = outputs.size(0)
        self.step_counter += 1
        
        # Apply softmax
        predictions = F.softmax(outputs, dim=-1)
        target_one_hot = F.one_hot(target, num_classes=outputs.size(-1)).float()
        
        # Calculate loss
        loss = -torch.sum(target_one_hot * torch.log(predictions + 1e-10)) / batch_size
        
        # Calculate accuracy
        with torch.no_grad():
            pred_digits = predictions.argmax(dim=-1)
            exact_match = (pred_digits == target).all(dim=1).float()
            match_rate = exact_match.mean().item()
            
            if print_details and self.step_counter % 1000 == 0:
                self._print_details(pred_digits, target, exact_match, loss, match_rate)
        
        return loss, match_rate
    
    def _print_details(self, pred_digits, target, exact_match, loss, match_rate):
        """Print detailed information"""
        print(f"\n=== Loss Calculation Details (Step: {self.step_counter}) ===")
        print("Predicted Sequences (First 10):", pred_digits[:10])
        print("\nActual Target Sequences (First 10):", target[:10])
        print("\nExact Match per Sequence (First 10):", exact_match[:10])
        print(f"\nCalculated Loss: {loss.item():.4f}")
        print(f"Calculated Accuracy: {match_rate:.4f}")
        print("="*40)

def create_addition_data(batch_size: int = 32, max_digits: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create addition dataset"""
    max_value = 10 ** max_digits - 1
    
    # Generate input numbers
    num1 = torch.randint(0, max_value // 2 + 1, (batch_size,))
    num2 = torch.randint(0, max_value // 2 + 1, (batch_size,))
    result = num1 + num2
    
    # Validation
    valid_mask = result < 10 ** max_digits
    while not valid_mask.all():
        invalid_idx = ~valid_mask
        num1[invalid_idx] = torch.randint(0, max_value // 2 + 1, (invalid_idx.sum(),))
        num2[invalid_idx] = torch.randint(0, max_value // 2 + 1, (invalid_idx.sum(),))
        result = num1 + num2
        valid_mask = result < 10 ** max_digits
    
    # Convert numbers to digit sequences
    num1_digits = _number_to_digits(num1, max_digits)
    num2_digits = _number_to_digits(num2, max_digits)
    result_digits = _number_to_digits(result, max_digits)
    
    # Create input sequence
    src = torch.zeros((batch_size, max_digits * 2 + 1), dtype=torch.long)
    src[:, :max_digits] = num1_digits
    src[:, max_digits] = 10  # '+' token
    src[:, max_digits+1:] = num2_digits
    
    return src, result_digits

def _number_to_digits(number: torch.Tensor, max_digits: int) -> torch.Tensor:
    """Convert a number to a sequence of digits"""
    return torch.tensor([[int(d) for d in str(n.item()).zfill(max_digits)] 
                        for n in number])

def _print_step_results(src, outputs, target, batch_size, max_digits):
    """Print prediction results per step"""
    predictions = outputs.argmax(dim=-1)
    
    print("\n=== Prediction Result Samples ===")
    for i in range(min(10, batch_size)):
        # Extract input numbers
        num1 = int(''.join(map(str, src[i, :max_digits].cpu().tolist())))
        num2 = int(''.join(map(str, src[i, max_digits+1:].cpu().tolist())))
        
        # Calculate the actual sum
        actual_sum = num1 + num2
        
        # Extract prediction
        pred_digits = predictions[i].cpu().tolist()
        pred_num = int(''.join(map(str, pred_digits)))
        
        # Print results
        match = "Correct" if pred_num == actual_sum else "Incorrect"
        print(f"{num1} + {num2} = {pred_num} (Prediction) / {actual_sum} (Result:) {match}")
    print("="*40)

def _print_epoch_summary(epoch: int, total_loss: float, total_accuracy: float, steps_per_epoch: int, optimizer) -> None:
    """Print epoch training results"""
    avg_loss = total_loss / steps_per_epoch
    avg_accuracy = total_accuracy / steps_per_epoch
    lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}, "
          f"Final Accuracy: {avg_accuracy:.4f}, "
          f"Learning Rate: {lr:.6f}")

def train_addition_task(
    config: TransformerConfig,
    num_epochs: int = 50,
    batch_size: int = 64,
    steps_per_epoch: int = 100,
    max_digits: int = 3,
    print_progress: bool = True
) -> Transformer:
    """Train the addition model"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    # )
    criterion = AdditionLoss().to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = total_accuracy = 0
        
        for step in range(steps_per_epoch):
            loss, accuracy = _train_step(
                model, criterion, optimizer, batch_size, max_digits, device,
                print_details=(step + 1) % steps_per_epoch == 0 and print_progress
            )
            
            total_loss += loss
            total_accuracy += accuracy
        
        if epoch % 5 == 0:
            _print_epoch_summary(epoch, total_loss, total_accuracy, steps_per_epoch, optimizer)
        # scheduler.step(total_loss / steps_per_epoch)
    save_model(model, config)
    
    return model

def _train_step(model, criterion, optimizer, batch_size, max_digits, device, print_details):
    """Perform a single training step"""
    src, tgt = create_addition_data(batch_size, max_digits)
    src, tgt = src.to(device), tgt.to(device)
    
    decoder_input = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    decoder_input = torch.cat([decoder_input, tgt[:, :-1]], dim=1)
    
    src_mask = create_pad_mask(src).to(device)
    tgt_mask = create_subsequent_mask(decoder_input.size(1)).to(device)
    
    optimizer.zero_grad()
    outputs = model(src, decoder_input, src_mask, tgt_mask)
    loss, accuracy = criterion(outputs, tgt, print_details)
    
    loss.backward()
    optimizer.step()
    
    # if print_details:
    #     _print_step_results(src, outputs, tgt, batch_size, max_digits)
    
    return loss.item(), accuracy

def test_addition(max_digits: int = 3) -> None:
    try:
        model, config = load_model()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Generate test data
        src, target = create_addition_data(batch_size=1, max_digits=max_digits)
        src, target = src.to(device), target.to(device)
        
        # Extract input numbers
        num1 = int(''.join(map(str, src[0, :max_digits].tolist())))
        num2 = int(''.join(map(str, src[0, max_digits+1:].tolist())))
        
        src_mask = create_pad_mask(src).to(device)
        
        with torch.no_grad():
            decoder_input = torch.zeros((1, 1), dtype=torch.long, device=device)
            output_sequence = []
            
            # Generate only up to max_digits
            for i in range(max_digits):
                tgt_mask = create_subsequent_mask(decoder_input.size(1)).to(device)
                output = model(src, decoder_input, src_mask, tgt_mask)
                next_token = output[:, -1:].argmax(dim=-1)
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
                output_sequence.append(next_token.item())
        
        predicted = int(''.join(map(str, output_sequence)))
        actual = num1 + num2
        
        print(f"\nAddition Test (Digits: {max_digits}):")
        print(f"{num1} + {num2} = {predicted} (Actual Answer: {actual})")
        print(f"Correct: {predicted == actual}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Train the model first.")


def main():
    if not os.path.exists(os.path.join(MODEL_PATH, MODEL_FILE)):
        print("=== Start Model Training ===")
        config = TransformerConfig()
        config.vocab_size = 11       # 0-9 digits + '+' symbol
        config.hidden_size = 128
        config.num_hidden_layers = 3
        config.num_attention_heads = 4
        config.intermediate_size = 256
        config.max_position_embeddings = 10
        
        model = train_addition_task(config, max_digits=3)
    
    print("\n=== 3-Digit Addition Test ===+")
    test_addition(max_digits=3)

if __name__ == "__main__":
    main()