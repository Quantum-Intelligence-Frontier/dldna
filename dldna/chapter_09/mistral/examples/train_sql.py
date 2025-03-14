import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import T5Tokenizer, get_cosine_schedule_with_warmup
import random
import os
from dldna.chapter_09.mistral.simple_mistral  import MistralConfig, MistralForCausalLM

# (1) Dataset class: Combine WikiSQL data into "question <sep> SQL<eos>" format and tokenize
class WikiSQLDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128):
        self.dataset = load_dataset("wikisql", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = sample['question']
        sql = sample['sql']['human_readable']
        # Construct the text in the format "question <sep> SQL<eos>".
        text = question + " <sep> " + sql + " <eos>"
        encodings = self.tokenizer(text,
                                   truncation=True,
                                   max_length=self.max_length,
                                   padding="max_length",
                                   return_tensors="pt")
        input_ids = encodings["input_ids"].squeeze(0)  # (max_length,)
        # In Causal LM training, input and label are configured the same.
        return input_ids, input_ids

# (2) Training function: Simple training loop, outputting average loss and current LR per epoch
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(dataloader):.4f}, LR: {current_lr:.6f}")

# (3) Generation function: Generate SQL for a question – Prompt is in the "question <sep> " format, terminate when EOS token is generated
def generate_sql(model, tokenizer, question, max_length, device, temperature=0.7):
    model.eval()
    # Construct the prompt in the format "question <sep> ".
    prompt = question + " <sep> "
    # Tokenize, removing padding and truncating to the actual prompt length.
    inputs = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    # Actual number of prompt tokens
    seq_len = input_ids.shape[1]
    generated_ids = input_ids

    with torch.no_grad():
        for _ in range(max_length - seq_len):
            outputs = model(input_ids=generated_ids)
            # Logits of the last token
            logits = outputs["logits"][:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            # If EOS or PAD token is generated, terminate
            if next_token_id.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
            # next_token_id is already in the shape [batch_size, 1], so remove unsqueeze(0).
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# (4) Main: Create model and tokenizer, dataset/dataloader, scheduler, train, evaluate, save final model
def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use T5Tokenizer as the tokenizer (use T5's vocab_size and pad/eos tokens)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # WikiSQL dataset (training: train, evaluation: validation)
    max_length = 128
    train_dataset = WikiSQLDataset("train", tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    valid_dataset = WikiSQLDataset("validation", tokenizer, max_length=max_length)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    # Model configuration: Use MistralConfig and MistralForCausalLM provided by simple_mistral.py
    # The model size is adjusted for educational purposes.
    config = MistralConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,     # num_attention_heads % num_key_value_heads == 0 must be true
        max_position_embeddings=max_length,
        sliding_window=max_length,
        use_cache=False,
        use_return_dict=True,
        pad_token_id=tokenizer.pad_token_id,  # Set the pad token id.
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = MistralForCausalLM(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 8  # Set the number of epochs small for the example
    total_training_steps = num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) // 5,
        num_training_steps=total_training_steps
    )
     # Added code: Output WikiSQL data samples
    print("=== WikiSQL Data Sample Output ===")
    sample_count = 3  # Number of examples to output
    for i in range(sample_count):
        input_ids, labels = train_dataset[i]
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Sample {i+1}: {decoded_text}")


    print("Start training...")
    train(model, train_loader, optimizer, scheduler, num_epochs, device)

    # Save the model: Save the final model to a file.
    torch.save(model.state_dict(), "final_nl2sql_model.pth")

    # Evaluation code part
    print("\n=== Evaluation Examples ===")
    for i, (input_ids, labels) in enumerate(valid_loader):
        if i >= 10:
            break
        # Keep special tokens with skip_special_tokens=False.
        full_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        # Unify the tokens "sep>" and "eos>" to "<sep>" and "<eos>" respectively.
        full_text = full_text.replace("sep>", "<sep>").replace("eos>", "<eos>")
        
        if "<sep>" in full_text:
            # Split based on the first <sep>, then join all subsequent parts to restore the complete SQL.
            parts = full_text.split("<sep>")
            question = parts[0].strip()
            target_sql = "<sep>".join(parts[1:]).strip()
            # If target_sql ends with "<eos>", remove it.
            if target_sql.endswith("<eos>"):
                target_sql = target_sql[:-len("<eos>")].strip()
        else:
            question = full_text.strip()
            target_sql = ""

        generated_sql = generate_sql(model, tokenizer, question, max_length, device, temperature=0.7)
        # If there is a "sep>" token in generated_sql, extract the part after that token to use.
        # if "sep>" in generated_sql:
        #     generated_sql = generated_sql.split("sep>", 1)[1].strip()

        print(f"Question: {question}")
        print(f"Target SQL: {target_sql}")
        print(f"Generated SQL: {generated_sql}\n")

if __name__ == "__main__":
    main()