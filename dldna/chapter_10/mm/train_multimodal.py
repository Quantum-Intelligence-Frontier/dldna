import torch
import torch.nn as nn
import torch.nn.functional as F

# from dldna.chapter_10.mm.cross_attention.v2 import CrossAttention # removed
import dldna.chapter_10.mm.cross_attention as ca  # import module
import argparse  # added
import time
import pandas as pd  # Added pandas for DataFrame usage
import warnings
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')


from dldna.chapter_10.mm.cross_attention.v0 import CrossAttention as v0
from dldna.chapter_10.mm.cross_attention.v1 import CrossAttention as v1
from dldna.chapter_10.mm.cross_attention.v2 import CrossAttention as v2
from dldna.chapter_10.mm.cross_attention.v3 import CrossAttention as v3
from dldna.chapter_10.mm.cross_attention.v4 import CrossAttention as v4
from dldna.chapter_10.mm.cross_attention.v5 import CrossAttention as v5
from dldna.chapter_10.mm.cross_attention.v6 import CrossAttention as v6
from dldna.chapter_10.mm.cross_attention.v7 import CrossAttention as v7
from dldna.chapter_10.mm.cross_attention.v8 import CrossAttention as v8
from dldna.chapter_10.mm.cross_attention.v9 import CrossAttention as v9
from dldna.chapter_10.mm.cross_attention.v10_1 import CrossAttention as v10_1
from dldna.chapter_10.mm.cross_attention.v10_2 import CrossAttention as v10_2
from dldna.chapter_10.mm.cross_attention.v10_3 import CrossAttention as v10_3
from dldna.chapter_10.mm.cross_attention.v10_4 import CrossAttention as v10_4
from dldna.chapter_10.mm.cross_attention.v10_5 import CrossAttention as v10_5
from dldna.chapter_10.mm.cross_attention.v10_6 import CrossAttention as v10_6
from dldna.chapter_10.mm.cross_attention.v11 import CrossAttention as v11

def get_cross_attention(version, config=None):
    if config is None:
        config = {}

    if version == 'v0':
        return v0(**config)
    elif version == 'v1':
        return v1(**config)
    elif version == 'v2':
        return v2(**config)
    elif version == 'v3':
        return v3(**config)
    elif version == 'v4':
        return v4(**config)
    elif version == 'v5':
        return v5(**config)
    elif version == 'v6':
        return v6(**config)
    elif version == 'v7':
        return v7(**config)
    elif version == 'v8':
        return v8(**config)
    elif version == 'v9':
        return v9(**config)
    elif version == 'v10_1':
        return v10_1(**config)
    elif version == 'v10_2':
        return v10_2(**config)
    elif version == 'v10_3':
        return v10_3(**config)
    elif version == 'v10_4':
        return v10_4(**config)
    elif version == 'v10_5':
        return v10_5(**config)
    elif version == 'v10_6':
        return v10_6(**config)
    elif version == 'v11':
        return v11(**config)
    else:
        raise ValueError(f"Invalid cross-attention version: {version}")
    

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.captions = []
        self.valid_items = []

        # Load image names and captions
        with open(caption_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                image_name = parts[0]
                caption = ','.join(parts[1:]).strip()

                # Check if the image file exists
                img_path = os.path.join(image_dir, image_name)
                if os.path.exists(img_path):
                    self.images.append(image_name)
                    self.captions.append(caption)
                    self.valid_items.append(True)
                else:
                    print(f"Image file not found: {img_path}")
                    self.valid_items.append(False)

        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        print(f"Total items: {len(self.valid_items)}")
        print(f"Valid items: {sum(self.valid_items)}")
        print(f"Invalid items: {len(self.valid_items) - sum(self.valid_items)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]
        tokens = self.tokenizer(caption, padding='max_length', truncation=True,
                              max_length=64, return_tensors="pt")

        return image, tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()


class ImageTextMatchingModel(nn.Module):
    def __init__(self, image_encoder_dim=2048, text_encoder_dim=768, projection_dim=256):
        super().__init__()
        self.image_encoder = ImageEncoder(image_encoder_dim, projection_dim)
        self.text_encoder = TextEncoder(text_encoder_dim, projection_dim)

        # The CrossAttention module is dynamically assigned in main().
        self.cross_attention = None #  CrossAttention(projection_dim)

        self.temp = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, image, input_ids, attention_mask):
        # Image and text encoding
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)

        # Apply bidirectional cross-attention
        image_attended, text_attended = self.cross_attention(
            image_features.unsqueeze(1),  # [batch_size, 1, projection_dim]
            text_features.unsqueeze(1)     # [batch_size, 1, projection_dim]
        )

        # Compress attention results
        image_features = image_attended.squeeze(1)
        text_features = text_attended.squeeze(1)

        # L2 normalization
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Calculate similarity
        logits = torch.matmul(image_features, text_features.t()) / self.temp
        return logits


class ImageEncoder(nn.Module):
    def __init__(self, encoder_dim, projection_dim):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.encoder.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, projection_dim)
            # nn.LayerNorm(projection_dim),  # Commented out
            # nn.Linear(projection_dim, projection_dim) # Commented out
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, encoder_dim, projection_dim):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, projection_dim)
            # nn.LayerNorm(projection_dim), # Commented out
            # nn.Linear(projection_dim, projection_dim) # Commented out
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.pooler_output
        x = self.projection(x)
        return x

def contrastive_loss(logits):
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_t = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_i + loss_t) / 2


def train(model, train_loader, val_loader, epochs=10, lr=1e-4, model_version='v0'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, epochs)

    best_contrast_ratio = -float('inf')
    train_loss_history = []
    val_loss_history = []
    accuracy_history = []
    contrast_ratio_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', mininterval=30.0):
            images, input_ids, attention_mask = [x.to(device) for x in batch]

            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask)
            loss = contrastive_loss(logits)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        scheduler.step()
        epoch_duration = time.time() - start_time  # Epoch duration


        # Validation and metric calculation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            all_logits = []

            for batch in val_loader:
                images, input_ids, attention_mask = [x.to(device) for x in batch]
                logits = model(images, input_ids, attention_mask)
                loss = contrastive_loss(logits)
                val_loss += loss.item()
                all_logits.append(logits)

            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)

            all_logits = torch.cat(all_logits, dim=0)

            labels = torch.arange(all_logits.size(0), device=device)
            probs = torch.nn.functional.softmax(all_logits, dim=1)
            predicted = probs.argmax(dim=1)
            accuracy = (predicted == labels).float().mean().item()
            accuracy_history.append(accuracy)

            diag_sim = torch.diagonal(all_logits).mean().item()
            off_diag_sim = (all_logits.sum() - torch.diagonal(all_logits).sum()) / (all_logits.numel() - all_logits.size(0))
            contrast_ratio = diag_sim / off_diag_sim if off_diag_sim != 0 else float('inf')
            contrast_ratio_history.append(contrast_ratio)

            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Accuracy = {accuracy:.4f}, Contrast Ratio = {contrast_ratio:.4f}, Time: {epoch_duration:.2f}s')

            # Save model (based on contrast ratio, including version information)
            # if contrast_ratio > best_contrast_ratio:
            #     best_contrast_ratio = contrast_ratio
            #     torch.save(model.state_dict(), f'best_model_{model_version}_epoch{epoch+1}.pth')
            #     print(f"Epoch {epoch+1}: Saved best model with Contrast Ratio = {best_contrast_ratio:.4f}")

        # Early stopping
        if epoch > 3 and val_loss_history[-1] > max(val_loss_history[-4:-1]):
            print("Early stopping triggered.")
            break

        # # Run evaluation_example for each epoch, using train_loader. evaluate_examples is commented out for now
        # evaluate_examples(model, train_loader, device)

    torch.save(model.state_dict(), f'model_final_{model_version}.pth')
    return train_loss_history, val_loss_history, accuracy_history, contrast_ratio_history, epoch_duration # Added return of training time


def evaluate_examples(model, train_loader, device):
    model.eval()
    with torch.no_grad():

        # Load image for zero-shot testing
        test_image = "./cat_resized.png"
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_image = Image.open(test_image).convert('RGB')
        test_image_tensor = transform(test_image).unsqueeze(0).to(device)

        # Test captions
        test_captions = [
            "A dog playing in the park",
            "A cat sleeping on a couch",
            "Children playing soccer",
            "A sunset over the ocean",
            "A person cooking in the kitchen"
        ]

        print("\n=== Zero-shot Testing ===")
        for caption in test_captions:
             tokens = train_loader.dataset.dataset.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )

             test_input_ids = tokens['input_ids'].to(device)
             test_attention_mask = tokens['attention_mask'].to(device)

             test_logits = model(test_image_tensor, test_input_ids, test_attention_mask)
             similarity = test_logits[0][0].item()
             print(f"Caption: {caption:<40} Similarity: {similarity:.4f}")



def run_training(model_versions, epochs=3, lr=1e-4, image_dir='./data/flickr8k/Images', caption_file='./data/flickr8k/captions.txt', batch_size=24, num_workers=2):
    """
    Trains multiple model versions and returns the results as a DataFrame.

    Args:
        model_versions (list): List of CrossAttention model versions to train.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        image_dir (str): Path to the image directory.
        caption_file (str): Path to the caption file.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for the data loader.

    Returns:
        pd.DataFrame: DataFrame containing training results for each model version.
    """

    results = []

    for model_version in model_versions:
        print(f"Training model version: {model_version}")

        # Configuration setup (if needed)
        config = {'dim': 256}  # Example config. May vary depending on the CrossAttention version.

        # Data transformation setup
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Dataset and DataLoader setup
        dataset = Flickr8kDataset(image_dir=image_dir, caption_file=caption_file, transform=transform)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)  # Added drop_last=True
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True) # Added drop_last=True

        # Model initialization
        model = ImageTextMatchingModel()

        # Dynamically load the CrossAttention module
        model.cross_attention = get_cross_attention(model_version, config=config)

        # Train the model and get results
        train_loss, val_loss, accuracy, contrast_ratio, total_time = train(model, train_loader, val_loader, epochs=epochs, lr=lr, model_version=model_version)


        # Store results
        result_dict = {
            'model_version': model_version,
            'final_train_loss': train_loss[-1],
            'final_val_loss': val_loss[-1],
            'final_accuracy': accuracy[-1],
            'best_contrast_ratio': max(contrast_ratio), # Record the best contrast ratio
            'total_training_time': total_time,
            # 'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad) # Number of trainable parameters
        }
        results.append(result_dict)
        print(f"Finished training model version: {model_version}")
        print("-" * 50)


    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    return df_results


def main():  # This main function is now for command-line execution.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, default='v2', help='Cross-attention model version (e.g., v2, v9)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    run_training(model_version=args.model_version, epochs=args.epochs, lr=args.lr) # Call run_training


if __name__ == '__main__':
    # model_versions = ['v0', 'v1']  # List of model versions to train
    model_versions = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10_1', 'v10_2', 'v10_3', 'v10_4', 'v10_5', 'v10_6', 'v11']
    epochs = 5
    lr = 1e-4

    results_df = run_training(model_versions, epochs=epochs, lr=lr) # Train multiple versions

    # Print results
    print("\nTraining Results:")
    # Print results in Markdown table format
    print(results_df.to_markdown(index=False))