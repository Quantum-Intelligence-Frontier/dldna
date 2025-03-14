import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math
from typing import Optional

from dldna.chapter_09.phi3.simple_phi3 import PhiMiniConfig, PhiMiniForCausalLM

# ---------------------------------------------------------------------
# 1. Complex Arithmetic Problem Data Generation and Tokenizer Related Functions
# ---------------------------------------------------------------------

def create_complex_arithmetic_data(num_samples: int, max_value: int) -> list:
    """
    Generates complex arithmetic expressions including two operators and optional parentheses.
    e.g., "12+7*3=33<eos>" or "(12+7)*3=57<eos>" or "12+(7*3)=33<eos>"
    """
    operations = ['+', '-', '*']
    data = []
    for _ in range(num_samples):
        a = random.randint(1, max_value)
        b = random.randint(1, max_value)
        c = random.randint(1, max_value)
        op1 = random.choice(operations)
        op2 = random.choice(operations)
        pattern = random.choice([0, 1, 2])
        if pattern == 0:
            # no parentheses
            expr = f"{a}{op1}{b}{op2}{c}"
        elif pattern == 1:
            # parentheses around the first two numbers
            expr = f"({a}{op1}{b}){op2}{c}"
        else:
            # parentheses around the last two numbers
            expr = f"{a}{op1}({b}{op2}{c})"
        try:
            result = str(eval(expr))
        except Exception as e:
            # If an error occurs during calculation, substitute with a simple expression
            expr = f"{a}{op1}{b}{op2}{c}"
            result = str(eval(expr))
        sample = f"{expr}={result}<eos>"
        data.append(sample)
    return data

def create_tokenizer() -> dict:
    """
    Adds '(' and ')' tokens to the existing tokenizer
    Numbers: '0'~'9' → indices 0~9
    Operators: '+', '-', '*' → 10, 11, 12
    '=' → 13
    <pad> → 14
    <eos> → 15
    and '(' → 16, ')' → 17
    """
    vocab = {}
    for i in range(10):
        vocab[str(i)] = i
    vocab['+'] = 10
    vocab['-'] = 11
    vocab['*'] = 12
    vocab['='] = 13
    vocab['<pad>'] = 14
    vocab['<eos>'] = 15
    vocab['('] = 16
    vocab[')'] = 17
    return vocab

def create_reverse_tokenizer(tokenizer: dict) -> dict:
    return {v: k for k, v in tokenizer.items()}

def tokenize_sample(sample: str, tokenizer: dict) -> list:
    """
    Tokenizes the string sample character by character, but recognizes special tokens like "<eos>" as a single token.
    """
    tokens = []
    i = 0
    while i < len(sample):
        if sample[i] == '<':  # Start of a special token
            j = sample.find('>', i)
            if j != -1:
                token = sample[i:j+1]
                if token in tokenizer:
                    tokens.append(token)
                    i = j + 1
                    continue
        tokens.append(sample[i])
        i += 1
    return tokens

# ---------------------------------------------------------------------
# 2. Dataset Class
# ---------------------------------------------------------------------
class ComplexArithmeticDataset(Dataset):
    def __init__(self, data: list, seq_length: int, tokenizer: dict):
        """
        :param data: List of complex arithmetic expression strings (e.g., "(12+7)*3=57<eos>")
        :param seq_length: Fixed sequence length (padded with <pad> tokens if necessary)
        :param tokenizer: Character-level tokenizer dictionary
        """
        self.data = data
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = tokenize_sample(text, self.tokenizer)
        if len(tokens) < self.seq_length:
            tokens = tokens + ['<pad>'] * (self.seq_length - len(tokens))
        else:
            tokens = tokens[:self.seq_length]
        token_ids = [self.tokenizer[token] for token in tokens]
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label_ids = torch.tensor(token_ids, dtype=torch.long)  # Causal LM: input == label
        return input_ids, label_ids

# ---------------------------------------------------------------------
# 3. Training Function and Text Generation Function
# ---------------------------------------------------------------------

def train(model, dataloader, optimizer, scheduler, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(dataloader):.4f}, LR: {current_lr:.6f}")

def generate_text(model, start_text, tokenizer, reverse_tokenizer, max_length, device, temperature=0.7):
    model.eval()
    token_ids = [tokenizer[token] for token in tokenize_sample(start_text, tokenizer)]
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = list(tokenize_sample(start_text, tokenizer))
    with torch.no_grad():
        for _ in range(max_length - len(token_ids)):
            outputs = model(input_ids=input_ids, return_dict=True)
            logits = outputs["logits"][:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            next_token = reverse_tokenizer[next_token_id]
            if next_token in ['<eos>', '<pad>']:
                break
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)
    return "".join(generated)

# ---------------------------------------------------------------------
# 4. Main Execution: Data Generation, Model Training, Saving, and Testing
# ---------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    num_samples = 10000   # Total number of data samples
    max_value = 20        # Maximum value of operands in arithmetic operations
    seq_length = 20       # Fixed sequence length including <eos> token
    batch_size = 16
    epochs = 20
    learning_rate = 1e-3

    # Data generation
    complex_data = create_complex_arithmetic_data(num_samples, max_value)
    print("Training data examples:")
    for i in range(5):
        print(f"Sample {i+1}: {complex_data[i]}")

    # Create tokenizer and reverse tokenizer
    tokenizer = create_tokenizer()
    reverse_tokenizer = create_reverse_tokenizer(tokenizer)
    updated_vocab_size = len(tokenizer)

    # Configure Dataset and DataLoader
    dataset = ComplexArithmeticDataset(complex_data, seq_length, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # PhiMini Model Configuration
    config = PhiMiniConfig(
        vocab_size=updated_vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,        # Note: key, value heads are also kept the same as 8 to ensure K=V=Q
        num_key_value_heads=8,
        max_position_embeddings=256,  # Context length is also slightly increased to handle longer inputs
        use_cache=True,
        use_return_dict=True,
    )
    config.pad_token_id = tokenizer["<pad>"]
    config.eos_token_id = tokenizer["<eos>"]

    # Create PhiMini For CausalLM Model
    model = PhiMiniForCausalLM(config).to(device)
    print("Total Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # weight tying (share weights between embedding and lm_head)
    model.lm_head.weight = model.transformer.embed_tokens.weight

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Model Training
    print("Start training...")
    train(model, dataloader, optimizer, scheduler, epochs, device)

    # Save Model
    save_path = "phimini_complex_math.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")

    # Load Saved Model (create a new model object before testing and load_state_dict)
    loaded_model = PhiMiniForCausalLM(config).to(device)
    loaded_model.load_state_dict(torch.load(save_path, map_location=device))
    loaded_model.eval()

    # Generate and Print Results with Test Set, Calculate Accuracy
    print("\nTest sample generation results:")
    test_samples = random.sample(complex_data, 50)
    correct_count = 0
    for sample in test_samples:
        prompt = sample.split('=')[0] + '='
        generated = generate_text(loaded_model, prompt, tokenizer, reverse_tokenizer, seq_length, device, temperature=0.1)  # Reduce temperature for testing
        answer = sample.split('=')[1].replace('<eos>', '')

        if generated.split('=')[1] == answer:
            correct_count += 1
        print(f"Prompt: '{prompt}' --> Generated result: '{generated}'  (Correct answer: {sample})")

    accuracy = (correct_count / len(test_samples)) * 100
    print(f"\nOverall accuracy: {accuracy:.2f}% ({correct_count}/{len(test_samples)})")