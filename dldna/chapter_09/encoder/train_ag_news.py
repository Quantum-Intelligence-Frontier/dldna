
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizerFast, T5Tokenizer  # Import different tokenizers
import os
from typing import Tuple, List, Dict
# Assuming efficient_encoder.py is in the same directory
from dldna.chapter_09.encoder.efficient_encoder_rope import TransformerConfig, TransformerEncoder  # Import v2
from dldna.chapter_09.encoder.efficient_encoder import TransformerConfig, TransformerEncoder as TransformerEncoderV1  # Import v1


MODEL_PATH = "saved_models"
# Use a base filename; we'll add the tokenizer name and version
MODEL_FILE_BASE = "efficient_transformer_agnews_encoder"
MAX_LENGTH = 128
BATCH_SIZE = 32


class AGNewsDataset(Dataset):
    def __init__(self, split="train", tokenizer_name="bert-base-uncased"):
        self.dataset = load_dataset("ag_news", split=split)
        # Select tokenizer based on name
        if tokenizer_name == "bert-base-uncased":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        elif tokenizer_name == "t5-small":  # Example of a different tokenizer
              self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        # Add other tokenizer options here (e.g., "roberta-base", "distilbert-base-uncased")
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']

        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

class EncoderClassifier(nn.Module):
    def __init__(self, config, version="v2"):
        super().__init__()
        if version == "v2":
          self.encoder = TransformerEncoder(config)  # Use the efficient encoder (v2)
        elif version == "v1":
          self.encoder = TransformerEncoderV1(config)
        else:
          raise ValueError("Invalid version. Choose 'v1' or 'v2'.")

        self.classifier = nn.Linear(config.hidden_size, 4) # AG News has 4 classes

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        # Use the output of the first token (CLS token in BERT) for classification
        return self.classifier(encoder_output[:, 0, :])


def save_model(model: EncoderClassifier, config: TransformerConfig, tokenizer_name: str, version: str) -> None:
    model_file = f"{MODEL_FILE_BASE}_{tokenizer_name}_{version}.pth"  # Include tokenizer and version
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': vars(config),  # Convert config object to dictionary
        'tokenizer_name': tokenizer_name,  # Save the tokenizer name
        'version': version, # Save Model Version
    }
    torch.save(save_dict, os.path.join(MODEL_PATH, model_file))
    print(f"Model saved to {os.path.join(MODEL_PATH, model_file)}")


def load_model(tokenizer_name: str, version: str) -> Tuple[EncoderClassifier, TransformerConfig]:
    model_file = f"{MODEL_FILE_BASE}_{tokenizer_name}_{version}.pth"
    model_path = os.path.join(MODEL_PATH, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")

    saved_dict = torch.load(model_path)

    # Recreate the config object from the saved dictionary
    config = TransformerConfig()
    for key, value in saved_dict['config'].items():
        setattr(config, key, value)
    # Ensure the loaded config uses the correct vocab size.
    if tokenizer_name == "bert-base-uncased":
        config.vocab_size = 30522
    elif tokenizer_name == "t5-small":
        config.vocab_size = 32100

    model = EncoderClassifier(config, version=saved_dict['version']) # Pass Version
    model.load_state_dict(saved_dict['model_state_dict'])

    return model, config


def train_agnews(config, tokenizer_name="bert-base-uncased", version="v2", verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    train_dataset = AGNewsDataset("train", tokenizer_name)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = EncoderClassifier(config, version=version).to(device)  # Pass version to the model


    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    n_epochs = 3
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # --- Masking fix ---
            # Get batch size from input_ids
            batch_size = input_ids.size(0)
            # 1. Ensure correct shape:
            attention_mask = attention_mask.view(batch_size, -1)

            # 2. Create the correct mask for F.scaled_dot_product_attention:
            attention_mask = attention_mask[:, None, None, :].bool()
            # --- End Masking fix ---


            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)

            # Compute loss and backpropagate
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if verbose: # Verbose
              if (batch_idx + 1) % 100 == 0:
                  print(f"Epoch [{epoch+1}/{n_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct/total:.2f}%")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        if verbose: # Verbose
          print(f"Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model



def test_agnews(tokenizer_name="bert-base-uncased", version="v2"):
    try:
        model, config = load_model(tokenizer_name, version)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        test_dataset = AGNewsDataset("test", tokenizer_name)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # --- Masking fix (same as in training) ---
                batch_size = input_ids.size(0) # Get batch_size here
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask = attention_mask[:, None, None, :].bool()
                # --- End Masking fix ---


                logits = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        # print(f"\nTest Accuracy ({tokenizer_name}): {accuracy:.2f}%") # Removed print
        return accuracy # Return accuracy

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first.")
        return 0.0


def predict_news(text: str, model: EncoderClassifier, config: TransformerConfig, device: torch.device, tokenizer_name="bert-base-uncased") -> Dict:
    """Predict the class of a news text"""
    # AG News class mapping
    label_map = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }

    # Initialize tokenizer
    if tokenizer_name == "bert-base-uncased":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == "t5-small":
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")
    

    # Preprocess the text
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # --- Masking fix (same as in training) ---
    attention_mask = attention_mask.view(1, -1)  # Assuming batch size 1 for prediction
    attention_mask = attention_mask[:, None, None, :].bool()
    # --- End Masking fix ---

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    return {
        'text': text,
        'predicted_class': label_map[predicted_class],
        'confidence': probs[0][predicted_class].item()
    }



def demo_news_classification(tokenizer_name="bert-base-uncased", version="v2"):
    """Demonstrate news classification with a few examples"""
    try:
        # Load model and config
        model, config = load_model(tokenizer_name, version)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)


        # Example news texts to test
        test_news = [
            "Oil prices surge as OPEC announces production cuts, impacting global markets",
            "Manchester United defeats Liverpool in dramatic Premier League clash",
            "SpaceX successfully launches new satellite constellation for global internet coverage",
            "New study shows promising results for Alzheimer's treatment",
            "Stock market hits all-time high amid positive economic indicators"
        ]
        print(f"\n=== News Classification Demo ({tokenizer_name}) ===")
        for news in test_news:
            result = predict_news(news, model, config, device, tokenizer_name)
            print("\nInput News:", result['text'])
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("-" * 80)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first.")


def train_and_test_all_versions(verbose=True):
    tokenizer_names = ["bert-base-uncased", "t5-small"]
    versions = ["v1", "v2"]
    results = {}

    for version in versions:
      results[version] = {}
      for tokenizer_name in tokenizer_names:
          model_file_path = os.path.join(MODEL_PATH, f"{MODEL_FILE_BASE}_{tokenizer_name}_{version}.pth")
          if not os.path.exists(model_file_path):
            print(f"=== Starting Model Training with {tokenizer_name} ({version}) ===")
            config = TransformerConfig()

            if tokenizer_name == "bert-base-uncased":
                config.vocab_size = 30522
            elif tokenizer_name == "t5-small":
                config.vocab_size = 32100

            if version == "v1":
              from dldna.chapter_09.encoder.efficient_encoder import TransformerEncoder as TransformerEncoderV1
              model = train_agnews(config, tokenizer_name, version, verbose)

            elif version == "v2":
              from dldna.chapter_09.encoder.efficient_encoder_rope import TransformerEncoder
              model = train_agnews(config, tokenizer_name, version, verbose)
            save_model(model, config, tokenizer_name, version) # Save with version

          test_accuracy = test_agnews(tokenizer_name, version) # Test and get accuracy
          results[version][tokenizer_name] = test_accuracy

          if verbose:
            print(f"\n=== Running Demo with Example News ({tokenizer_name}, {version}) ===")
            demo_news_classification(tokenizer_name, version) # Pass Version

    # Print results table
    print("\n=== Results ===")
    print("| Model Version | Tokenizer          | Test Accuracy |")
    print("|---------------|--------------------|---------------|")
    for version, tokenizer_results in results.items():
        for tokenizer, accuracy in tokenizer_results.items():
            print(f"| {version}           | {tokenizer:<18} | {accuracy:.2f}%       |")

def main():
    train_and_test_all_versions(verbose=True)

if __name__ == "__main__":
    main()