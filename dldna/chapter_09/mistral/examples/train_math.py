import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

from dldna.chapter_09.mistral.simple_mistral  import MistralConfig, MistralForCausalLM

# Data generation: Arithmetic expression evaluation (add EOS token)
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
        # Add EOS token.
        sample = f"{expr}={result}<eos>"
        data.append(sample)
    return data

# Create a character-level tokenizer (including digits, operators, and '=')
def create_tokenizer():
    vocab = {}
    # '0'-'9'
    for i in range(10):
        vocab[str(i)] = i
    # Operators: '+', '-', '*'
    vocab['+'] = 10
    vocab['-'] = 11
    vocab['*'] = 12
    # '='
    vocab['='] = 13
    # pad token
    vocab['<pad>'] = 14
    # eos token
    vocab['<eos>'] = 15
    return vocab

def create_reverse_tokenizer(tokenizer):
    return {v: k for k, v in tokenizer.items()}

# Tokenization function: Treat special tokens (<eos>, etc.) as a single token
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
            # If there's no closing '>', just process one character
        # Process normal characters one by one
        tokens.append(sample[i])
        i += 1
    return tokens

# Dataset class: Each sample is padded to a fixed length (seq_length)
class ArithmeticDataset(Dataset):
    def __init__(self, data, seq_length, tokenizer):
        """
        :param data: List of arithmetic expression strings (e.g., "12+7=19<eos>")
        :param seq_length: Fixed sequence length (shorter expressions are filled with <pad> tokens)
        :param tokenizer: Character-level tokenizer dictionary
        """
        self.data = data
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        string = self.data[idx]
        # custom tokenize: special tokens are treated as one token
        tokens = tokenize_sample(string, self.tokenizer)
        # If the string length is shorter than seq_length, fill it with <pad> tokens.
        if len(tokens) < self.seq_length:
            tokens = tokens + ['<pad>'] * (self.seq_length - len(tokens))
        else:
            tokens = tokens[:self.seq_length]
        token_ids = [self.tokenizer[token] for token in tokens]
        # In Causal LM training, the input and label are configured the same.
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label_ids = torch.tensor(token_ids, dtype=torch.long)
        return input_ids, label_ids

# Training function
def train(model, dataloader, optimizer, scheduler, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()  # Update learning rate after each epoch
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(dataloader):.4f}, LR: {current_lr:.6f}")

# Text generation function: Terminate when the EOS token is generated after the prompt
def generate_text(model, start_text, tokenizer, reverse_tokenizer, max_length, device, temperature=0.7):
    model.eval()
    # Convert prompt using custom tokenization function
    token_ids = [tokenizer[token] for token in tokenize_sample(start_text, tokenizer)]
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = list(tokenize_sample(start_text, tokenizer))
    with torch.no_grad():
        for _ in range(max_length - len(token_ids)):
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"][:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            next_token = reverse_tokenizer[next_token_id]
            # Terminate if it's <eos> or <pad>
            if next_token in ['<eos>', '<pad>']:
                break
            generated.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)],
                dim=1
            )
    return "".join(generated)

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameter settings
    num_samples = 10000   # Total number of samples in the dataset
    max_value = 20       # Maximum value of operands
    seq_length = 20      # Fixed sequence length including EOS token (e.g., 20)
    batch_size = 16
    epochs = 20
    learning_rate = 1e-3

    # Data generation (including EOS token) and output training data examples
    arithmetic_data = create_arithmetic_data(num_samples, max_value)
    print("Training data examples:")
    for i in range(10):
        print(f"Sample {i+1}: {arithmetic_data[i]}")

    # Create tokenizer
    tokenizer = create_tokenizer()
    reverse_tokenizer = create_reverse_tokenizer(tokenizer)
    updated_vocab_size = len(tokenizer)

    # Configure Dataset and DataLoader
    dataset = ArithmeticDataset(arithmetic_data, seq_length, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config = MistralConfig(
        vocab_size=updated_vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=128,
        sliding_window=seq_length,
        use_cache=False,
        use_return_dict=True,
        pad_token_id=tokenizer["<pad>"]  # Set the pad token id here.
    )
    config.eos_token_id = tokenizer["<eos>"]  # Also update the eos token

    model = MistralForCausalLM(config).to(device)

    # weight tying (share weights between embedding and lm_head)
    tie_weights = True
    if tie_weights:
        model.lm_head.weight = model.model.embed_tokens.weight

    # Create optimizer and add cosine annealing scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Start training
    print("Start training...")
    train(model, dataloader, optimizer, scheduler, epochs, device)

    # Evaluation: Output 10 random evaluation samples (terminate generation if EOS is included in the prompt)
    print("\nEvaluation data examples:")
    for i in range(10):
        sample = random.choice(arithmetic_data)
        # Use the part before '=' as a prompt in the entire expression, e.g., "12+7=19<eos>" ("12+7=")
        prompt = sample.split('=')[0] + '='
        generated = generate_text(model, prompt, tokenizer, reverse_tokenizer, max_length=seq_length, device=device)
        print(f"Generated result for prompt '{prompt}': {generated} (Original data: {sample})")