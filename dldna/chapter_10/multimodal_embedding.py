import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.captions = []

        with open(caption_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue

                image_name = parts[0]
                caption = ','.join(parts[1:]).strip()
                self.images.append(image_name)
                self.captions.append(caption)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Image file not found: {img_name}")
            image = Image.new('RGB', (224, 224), color='black')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]
        tokens = self.tokenizer(caption, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        return image, tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)


class CrossModalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_proj = nn.Linear(config.image_dim, config.hidden_dim)
        self.text_proj = nn.Linear(config.text_dim, config.hidden_dim)
        self.attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True)

    def forward(self, image_features, text_features):
        image_proj = self.image_proj(image_features)
        text_proj = self.text_proj(text_features)
        attn_output, _ = self.attention(text_proj, image_proj, image_proj)  # query, key, value
        return attn_output

class MultimodalEmbedding(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.image_encoder = models.resnet18(pretrained=True)
        self.image_encoder.fc = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Sequential(
            nn.Linear(768, embedding_dim),  # BERT output dimension is 768
            nn.LayerNorm(embedding_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        return self.image_encoder(image)

    def encode_text(self, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask)[0][:, 0, :]  # [CLS] token, keep batch dim
        return self.text_projection(text_features)

    def forward(self, image, input_ids, attention_mask):
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)
        # print("image feature:", image_features.shape)
        # print("text feature:", text_features.shape)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print("image normalized feature:", image_features.shape)
        # print("text normalized feature:", text_features.shape)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.transpose(-1, -2)
        # print("logits:", logits.shape)

        return logits   # Return a single value


def contrastive_loss(logits): # removed enhanced_similarity
    labels = torch.arange(logits.size(0), device=logits.device) # Use logits.size(0)

    # Image-to-text and text-to-image contrastive loss
    img_txt_loss = nn.CrossEntropyLoss()(logits, labels)
    txt_img_loss = nn.CrossEntropyLoss()(logits.T, labels)

    # Average loss
    return (img_txt_loss + txt_img_loss) / 2


def train_multimodal_embedding(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        running_loss = 0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"), 1):
            images, input_ids, attention_mask = [item.to(device) for item in batch]

            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask) # Receive only one logits

            # Use improved loss function
            loss = contrastive_loss(logits)  # Pass only one logits
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for batch in val_loader:
                images, input_ids, attention_mask = [item.to(device) for item in batch]
                logits = model(images, input_ids, attention_mask)
                loss = contrastive_loss(logits)
                val_loss += loss.item()

        avg_val_loss = val_loss / num_val_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")

        # Save best model (based on validation loss)
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), 'multimodal_embedding_model.pth')
        #     print(f"Epoch {epoch+1}: Saved best model with Validation Loss = {avg_val_loss:.4f}")


def display_image(image_tensor):
    """Args:
        image_tensor (torch.Tensor): Image tensor of shape (C, H, W)
    """
    # Move to CPU and clone
    img = image_tensor.cpu().clone()

    # If it's a 3-channel (RGB) image and the value range is not , but rather (e.g., normalized)
    # Restore to range (apply commonly used normalization values for Flickr images)
    if img.ndim == 3 and img.shape[0] == 3 and (img.min() < 0 or img.max() > 1):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = img * std + mean  # unnormalize

    # Clamp to range
    img = torch.clamp(img, 0, 1)

    # Convert torch tensor to PIL image and display (perform minimal conversion)
    pil_img = to_pil_image(img)
    plt.imshow(pil_img)
    plt.axis('off')
    plt.show()


def generate_example(model_path, image_dir, caption_file):
    """
    Loads a saved model, retrieves samples from the validation dataset,
    and prints the most similar captions for an image and the most similar images for a caption.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the model
    model = MultimodalEmbedding()  # Define model structure
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Evaluation mode

    # 2. Prepare the data loader (validation dataset)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = Flickr8kDataset(image_dir, caption_file, transform=transform)
    _, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # Don't shuffle.

    # 3. Get a sample from the validation dataset
    with torch.no_grad(): # No gradient calculation needed.
        images, input_ids, attention_mask = next(iter(val_loader))
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 4. Calculate similarities using the model
        logits = model(images, input_ids, attention_mask) # (batch, batch)
        image_features = model.encode_image(images)
        text_features = model.encode_text(input_ids, attention_mask)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        probs = torch.softmax(logits, dim=1) # Image -> Text similarity
        probs_t2i = torch.softmax(logits.t(), dim=1) # Text -> Image similarity

        # 5. Find the most similar captions for the first image
        image_idx = 0
        topk_values, topk_indices = torch.topk(probs[image_idx], 5) # Top 5 captions

        print(f"Image {image_idx}:")
        # Display the image (in Jupyter Notebook)
        display_image(images[image_idx])

        n_top = 3

        print(f"\nTop {n_top} Captions (Image -> Text):")
        for i in range(n_top):
          caption_idx = topk_indices[i]
          # Convert tokens back to text
          caption = dataset.tokenizer.decode(input_ids[caption_idx].cpu().numpy(), skip_special_tokens=True)
          print(f"  - {caption} (prob: {topk_values[i]:.4f})")

        # 6. Find the most similar images for the first caption.
        caption_idx = 0
        topk_values_t2i, topk_indices_t2i = torch.topk(probs_t2i[caption_idx], 5)  # Top 5 images
        caption = dataset.tokenizer.decode(input_ids[caption_idx].cpu().numpy(), skip_special_tokens=True)
        print(f"\nCaption: {caption}")
        print(f"\nTop {n_top} Images (Text -> Image):")

        for i in range(n_top):
            image_idx = topk_indices_t2i[i]
            print(f" - Image {image_idx} (prob: {topk_values_t2i[i]:.4f})")
            display_image(images[image_idx])