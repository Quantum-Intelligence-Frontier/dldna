import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math

# Import Phi-Mini model related classes
from dldna.chapter_09.phi3.simple_phi3 import PhiMiniConfig, PhiMiniForCausalLM

# ------------------------------
# Data Generation and Tokenizer Related Functions
# ------------------------------

def create_arithmetic_data(num_samples, max_value):
    """
    Generates arithmetic expressions and returns them as strings in the form 'expression=result<eos>'.
    e.g., "12+7=19<eos>"
    """
    operations = ['+', '-', '*']
    data = []
    for _ in range(num_samples):
        num1 = random.randint(1, max_value)
        num2 = random.randint(1, max_value)
        op = random.choice(operations)
        expr = f"{num1}{op}{num2}"
        result = str(eval(expr))
        sample = f"{expr}={result}<eos>"  # Add EOS token
        data.append(sample)
    return data

def create_tokenizer():
    vocab = {}
    # '0'-'9'
    for i in range(10):
        vocab[str(i)] = i
    # Operators: '+', '-', '*'
    vocab['+'] = 10
    vocab['-'] = 11
    vocab['*'] = 12
    # '=' symbol
    vocab['='] = 13
    # pad token
    vocab['<pad>'] = 14
    # eos token
    vocab['<eos>'] = 15
    return vocab

def create_reverse_tokenizer(tokenizer):
    return {v: k for k, v in tokenizer.items()}

def tokenize_sample(sample, tokenizer):
    """
    Converts the sample string into a list of tokens.
    If there is a special token like "<eos>", it is recognized as one token.
    Otherwise, it is separated character by character.
    """
    tokens = []
    i = 0
    while i < len(sample):
        if sample[i] == '<':  # If it's the start of a special token
            j = sample.find('>', i)
            if j != -1:
                token = sample[i:j+1]  # e.g., "<eos>"
                if token in tokenizer:
                    tokens.append(token)
                    i = j + 1
                    continue
        tokens.append(sample[i])
        i += 1
    return tokens

# ------------------------------
# Dataset Class
# ------------------------------

class ArithmeticDataset(Dataset):
    def __init__(self, data, seq_length, tokenizer):
        """
        :param data: List of arithmetic expression strings (e.g., "12+7=19<eos>")
        :param seq_length: Fixed sequence length (shorter expressions are padded with <pad> tokens)
        :param tokenizer: Character-level tokenizer dictionary
        """
        self.data = data
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        string = self.data[idx]
        tokens = tokenize_sample(string, self.tokenizer)
        # Pad with <pad> tokens if shorter than sequence length.
        if len(tokens) < self.seq_length:
            tokens = tokens + ['<pad>'] * (self.seq_length - len(tokens))
        else:
            tokens = tokens[:self.seq_length]
        token_ids = [self.tokenizer[token] for token in tokens]
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label_ids = torch.tensor(token_ids, dtype=torch.long)  # Causal LM: Input and label are the same
        return input_ids, label_ids

# ------------------------------
# Training Function
# ------------------------------

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

        scheduler.step()  # Update learning rate after each epoch
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(dataloader):.4f}, LR: {current_lr:.6f}")

# ------------------------------
# Text Generation Function
# ------------------------------

def generate_text(model, start_text, tokenizer, reverse_tokenizer, max_length, device, temperature=0.7):
    model.eval()
    # Convert prompt using custom tokenization function
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
            # Terminate if <eos> or <pad> token is generated
            if next_token in ['<eos>', '<pad>']:
                break
            generated.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)],
                dim=1
            )
    return "".join(generated)

# ------------------------------
# Main Execution: Data Generation, Model Configuration, Training, and Evaluation
# ------------------------------

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameter settings
    num_samples = 10000   # Total number of data samples
    max_value = 20        # Maximum value of operands in arithmetic operations
    seq_length = 20       # Fixed sequence length including <eos> token
    batch_size = 16
    epochs = 20
    learning_rate = 1e-3

    # Data generation and sample output
    arithmetic_data = create_arithmetic_data(num_samples, max_value)
    print("Training data examples:")
    for i in range(10):
        print(f"Sample {i+1}: {arithmetic_data[i]}")

    # Tokenizer creation
    tokenizer = create_tokenizer()
    reverse_tokenizer = create_reverse_tokenizer(tokenizer)
    updated_vocab_size = len(tokenizer)

    # Dataset and DataLoader configuration
    dataset = ArithmeticDataset(arithmetic_data, seq_length, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Phi-Mini model settings
    config = PhiMiniConfig(
        vocab_size=updated_vocab_size,
        hidden_size=64,              # Small model size for experimentation
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=8,        # K=V=Q
        max_position_embeddings=128,
        use_cache=False,
        use_return_dict=True,
    )


    config.pad_token_id = tokenizer["<pad>"]
    config.eos_token_id = tokenizer["<eos>"]

    # Create Phi-Mini model
    model = PhiMiniForCausalLM(config).to(device)

    print("Total Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # weight tying (share weights between embedding and lm_head)
    tie_weights = True
    if tie_weights:
        model.lm_head.weight = model.transformer.embed_tokens.weight

    # Optimizer and learning rate scheduler configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Model training
    print("Start training...")
    train(model, dataloader, optimizer, scheduler, epochs, device)

    # Evaluation: Text generation for 10 random samples
    print("\nEvaluation data examples:")
    for i in range(10):
        sample = random.choice(arithmetic_data)
        # Prompt: Use the expression up to "=" (e.g., "12+7=")
        prompt = sample.split('=')[0] + '='
        generated = generate_text(model, prompt, tokenizer, reverse_tokenizer, max_length=seq_length, device=device)
        print(f"Prompt '{prompt}' Generated Result: {generated} (Original Data: {sample})")