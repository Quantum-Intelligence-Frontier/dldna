import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dldna.chapter_09.mistral.simple_mistral import MistralConfig, MistralForCausalLM

# 1. Define a simple dataset
class SimpleDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx : idx + self.seq_length], dtype=torch.long)
        return seq, seq  # Same sequence for input and labels, labels are automatically shifted in the model's forward pass.

def create_simple_data(vocab_size: int, num_examples: int, seq_length: int) -> list:
    """
    Generates a sequentially increasing number sequence.
    """
    base_seq = list(range(vocab_size))  # Generate numbers from 0 to vocab_size-1
    data = base_seq * ((num_examples // len(base_seq)) + 1)  # Repeat enough times to create at least num_examples
    return data[:num_examples]  # Truncate to num_examples length.

# 2. Training function
def train(model, dataloader, optimizer, epochs, device):
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

            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

# 3. Text generation function (maintaining temperature and sampling method)
def generate_text(model, start_text, tokenizer_vocab, max_length, device, temperature=0.7):
    model.eval()
    try:
        input_ids = [tokenizer_vocab[token] for token in start_text]
    except KeyError as e:
        raise ValueError(f"Token {e} does not exist in tokenizer_vocab.")
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated_tokens = start_text[:]

    with torch.no_grad():
        for _ in range(max_length - len(start_text)):
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"][:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            # reverse lookup: token id -> token string
            next_token = None
            for token, tid in tokenizer_vocab.items():
                if tid == next_token_id:
                    next_token = token
                    break
            if next_token is None or next_token == "<eos>":
                break

            generated_tokens.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)],
                dim=1
            )
    return generated_tokens

# --- Main ---
if __name__ == "__main__":
    # Hyperparameter settings
    base_vocab_size = 50    # Original vocab_size before the EOS token
    seq_length = 10         # Sequence length of each training sample
    batch_size = 8
    epochs = 5
    learning_rate = 5e-3
    num_train_examples = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Create tokenizer (string token -> token id)
    tokenizer_vocab = {str(i): i for i in range(base_vocab_size)}
    tokenizer_vocab["<eos>"] = base_vocab_size
    updated_vocab_size = base_vocab_size + 1

    # 2) Model configuration: Apply the updated vocab_size and set sliding_window to seq_length
    config = MistralConfig(
        vocab_size=updated_vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        sliding_window=seq_length,  # Set to the same as the sequence length
        use_cache=False  # Do not use cache during training
    )
    config.eos_token_id = tokenizer_vocab["<eos>"]

    # (Optional) Set up weight tying between embedding and lm_head -> Can help reproduce sequential patterns.
    tie_weights = True

    # 3) Create model and Optimizer
    model = MistralForCausalLM(config).to(device)
    if tie_weights:
        model.lm_head.weight = model.model.embed_tokens.weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 4) Data generation and DataLoader preparation
    train_data = create_simple_data(updated_vocab_size, num_train_examples, seq_length)
    train_dataset = SimpleDataset(train_data, seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- For debugging: Output some data before training ---
    print("Sample data before training (input sequence -> label sequence):")
    for i in range(2):
        input_seq, label_seq = train_dataset[i]
        print(f"Sample {i+1}: {input_seq.tolist()} -> {label_seq.tolist()}")

    # 5) Start training
    print("Start training...")
    train(model, train_dataloader, optimizer, epochs, device)

    # 6) Text generation example
    print("Generating text starting with tokens ['1', '2', '3']:")
    start_text = ["1", "2", "3"]
    generated = generate_text(model, start_text, tokenizer_vocab, max_length=20, device=device)
    print("Generated text:", " ".join(generated))

    print("Generating text starting with tokens ['40', '41', '42']:")
    start_text = ["40", "41", "42"]
    generated = generate_text(model, start_text, tokenizer_vocab, max_length=20, device=device)
    print("Generated text:", " ".join(generated))