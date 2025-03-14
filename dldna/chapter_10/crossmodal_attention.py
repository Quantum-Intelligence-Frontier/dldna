# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms
# from transformers import BertModel, BertTokenizer
# from PIL import Image
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from torchvision.transforms.functional import to_pil_image
# from collections import namedtuple

# class Flickr8kDataset(Dataset):
#     def __init__(self, image_dir, caption_file, transform=None):
#         self.image_dir = image_dir
#         self.transform = transform
#         self.images = []
#         self.captions = []

#         with open(caption_file, 'r') as f:
#             for line in f:
#                 parts = line.strip().split(',')
#                 if len(parts) < 2:
#                     continue

#                 image_name = parts[0]
#                 caption = ','.join(parts[1:]).strip()
#                 self.images.append(image_name)
#                 self.captions.append(caption)

#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.image_dir, self.images[idx])
#         try:
#             image = Image.open(img_name).convert('RGB')
#         except FileNotFoundError:
#             print(f"Image file not found: {img_name}")
#             image = Image.new('RGB', (224, 224), color='black')
#         except Exception as e:
#             print(f"Error loading image {img_name}: {e}")
#             image = Image.new('RGB', (224, 224), color='black')

#         if self.transform:
#             image = self.transform(image)

#         caption = self.captions[idx]
#         tokens = self.tokenizer(caption, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
#         return image, tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)


# class CrossModalAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.image_proj = nn.Linear(config.image_dim, config.hidden_dim)
#         self.text_proj = nn.Linear(config.text_dim, config.hidden_dim)
#         self.attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True)

#     def forward(self, image_features, text_features):
#         image_proj = self.image_proj(image_features)
#         text_proj = self.text_proj(text_features)
#         attn_output, _ = self.attention(text_proj, image_proj, image_proj)  # query, key, value
#         return attn_output

# class HierarchicalCrossModalAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.local_image_attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True)
#         self.local_text_attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True)
#         self.image_to_text_attention = CrossModalAttention(config)
#         self.text_to_image_attention = CrossModalAttention(config)
#         self.output_layer = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

#     def forward(self, image_features, text_features):
#         local_image = self.local_image_attention(image_features, image_features, image_features)[0]
#         local_text = self.local_text_attention(text_features, text_features, text_features)[0]

#         image_attended_text = self.image_to_text_attention(image_features, local_text)
#         text_attended_image = self.text_to_image_attention(text_features, local_image)

#         combined_features = torch.cat([image_attended_text, text_attended_image], dim=-1)  # Concatenate
#         output = self.output_layer(combined_features)
#         return output


# class CrossModalEmbedding(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.image_encoder = models.resnet18(pretrained=True)
#         self.image_encoder.fc = nn.Sequential(
#             nn.Linear(512, config.embedding_dim),
#             nn.LayerNorm(config.embedding_dim)
#         )

#         self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
#         self.text_projection = nn.Sequential(
#             nn.Linear(768, config.embedding_dim),  # BERT output dimension is 768
#             nn.LayerNorm(config.embedding_dim)
#         )
#         # cross-modal attention 설정
#         self.cross_modal_attention = HierarchicalCrossModalAttention(config)
#         # 마지막 projection layer
#         self.final_projection = nn.Linear(config.embedding_dim, config.embedding_dim)  # Add this line

#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#     def encode_image(self, image):
#         return self.image_encoder(image)

#     def encode_text(self, input_ids, attention_mask):
#         text_features = self.text_encoder(input_ids, attention_mask)[0][:, 0, :]  # [CLS] token, keep batch dim
#         return self.text_projection(text_features)

#     def forward(self, image, input_ids, attention_mask):
#         image_features = self.encode_image(image)  # (B, D)
#         text_features = self.encode_text(input_ids, attention_mask)  # (B, D)

#         # Normalize before attention (important for contrastive learning)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)


#         # 시퀀스 차원 추가
#         image_features = image_features.unsqueeze(1)  # (B, 1, D)
#         text_features = text_features.unsqueeze(1)  # (B, 1, D)

#         combined_features = self.cross_modal_attention(image_features, text_features) # (B, 1, D)

#         # Final projection and normalization
#         combined_features = self.final_projection(combined_features)  # Add this line
#         combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

#         pooled_features = combined_features.mean(dim=1) # (B, D)

#         #  cross-modal attention의 출력으로 유사도 계산.
#         # enhanced_similarity = self.logit_scale.exp() * pooled_features @ pooled_features.T
#         logits = self.logit_scale.exp() * pooled_features @ pooled_features.T

#         return logits


# def contrastive_loss(logits):
#     batch_size = logits.size(0)
#     labels = torch.arange(batch_size, device=logits.device)

#     # 이미지→텍스트, 텍스트→이미지 대조 손실
#     img_txt_loss = nn.CrossEntropyLoss()(logits, labels)
#     txt_img_loss = nn.CrossEntropyLoss()(logits.T, labels)
#     return (img_txt_loss + txt_img_loss) / 2



# def train_crossmodal_embedding(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     best_val_loss = float('inf')

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         running_loss = 0
#         num_batches = len(train_loader)

#         for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"), 1):
#             images, input_ids, attention_mask = [item.to(device) for item in batch]

#             optimizer.zero_grad()
#             logits = model(images, input_ids, attention_mask)

#             # 개선된 손실 함수 사용
#             loss = contrastive_loss(logits)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             running_loss += loss.item()
        #    # 100 배치마다 중간 손실 보고
        #     if batch_idx % 100 == 0:
        #         print(f"Batch {batch_idx}/{num_batches}, Loss: {running_loss/100:.4f}")
        #         running_loss = 0

#         avg_loss = total_loss / num_batches
#         print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")

#         # Validation 진행
#         model.eval()
#         val_loss = 0
#         num_val_batches = len(val_loader)
#         with torch.no_grad():
#             for batch in val_loader:
#                 images, input_ids, attention_mask = [item.to(device) for item in batch]
#                 logits = model(images, input_ids, attention_mask)
#                 loss = contrastive_loss(logits)
#                 val_loss += loss.item()

#         avg_val_loss = val_loss / num_val_batches
#         print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")

#         # 최적 모델 저장 (검증 손실 기준)
#         # if avg_val_loss < best_val_loss:
#         #     best_val_loss = avg_val_loss
#         #     torch.save(model.state_dict(), 'best_crossmodal_embedding_model.pth')
#         #     print(f"Epoch {epoch+1}: Saved best model with Validation Loss = {avg_val_loss:.4f}")


# def generate_example(model_path, image_dir, caption_file):
#     """
#     Loads a saved model, retrieves samples from the validation dataset,
#     and prints the most similar captions for an image and the most similar images for a caption.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 1. Load the model
#     # Config 설정 (모델에 맞게)
#     config = namedtuple('Config', ['embedding_dim', 'image_dim', 'text_dim', 'hidden_dim', 'num_heads'])(
#       embedding_dim=512, image_dim=512, text_dim=768, hidden_dim=512, num_heads=8  # BERT text_dim
#     )
#     model = CrossModalEmbedding(config)  # Define model structure
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()  # Evaluation mode

#     # 2. Prepare the data loader (validation dataset)
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     dataset = Flickr8kDataset(image_dir, caption_file, transform=transform)
#     _, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # Don't shuffle.

#     # 3. Get a sample from the validation dataset
#     with torch.no_grad(): # No gradient calculation needed.
#         images, input_ids, attention_mask = next(iter(val_loader))
#         images = images.to(device)
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)

#         # 4. Calculate similarities using the model
#         logits = model(images, input_ids, attention_mask)  # (batch, batch)

#         probs = torch.softmax(logits, dim=1) # Image -> Text similarity
#         probs_t2i = torch.softmax(logits.t(), dim=1) # Text -> Image similarity

#         # 5. Find the most similar captions for the first image
#         image_idx = 0
#         topk_values, topk_indices = torch.topk(probs[image_idx], 5) # Top 5 captions

#         print(f"Image {image_idx}:")
#         # Display the image (in Jupyter Notebook)
#         display_image(images[image_idx])

#         n_top = 3

#         print(f"\nTop {n_top} Captions (Image -> Text):")
#         for i in range(n_top):
#             caption_idx = topk_indices[i]
#             # Convert tokens back to text
#             caption = dataset.tokenizer.decode(input_ids[caption_idx].cpu().numpy(), skip_special_tokens=True)
#             print(f"  - {caption} (prob: {topk_values[i]:.4f})")

#         # 6. Find the most similar images for the first caption.
#         caption_idx = 0
#         topk_values_t2i, topk_indices_t2i = torch.topk(probs_t2i[caption_idx], 5)  # Top 5 images
#         caption = dataset.tokenizer.decode(input_ids[caption_idx].cpu().numpy(), skip_special_tokens=True)
#         print(f"\nCaption: {caption}")
#         print(f"\nTop {n_top} Images (Text -> Image):")

#         for i in range(n_top):
#             image_idx = topk_indices_t2i[i]
#             print(f" - Image {image_idx} (prob: {topk_values_t2i[i]:.4f})")
#             display_image(images[image_idx])



# def display_image(image_tensor):
#     """
#     Args:
#         image_tensor (torch.Tensor): Image tensor of shape (C, H, W)
#     """
#     # Move to CPU and clone
#     img = image_tensor.cpu().clone()

#     # If it's a 3-channel (RGB) image and the value range is not [0, 1], but rather (e.g., normalized)
#     # Restore to range [0, 1] (apply commonly used normalization values for Flickr images)
#     if img.ndim == 3 and img.shape[0] == 3 and (img.min() < 0 or img.max() > 1):
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
#         img = img * std + mean  # unnormalize

#     # Clamp to range [0, 1]
#     img = torch.clamp(img, 0, 1)

#     # Convert torch tensor to PIL image and display (perform minimal conversion)
#     pil_img = to_pil_image(img)
#     plt.imshow(pil_img)
#     plt.axis('off')
#     plt.show()


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
from collections import namedtuple

# -----------------------------------------------------------
# 1. Flickr8k Dataset Definition
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# 2. CrossModalEmbedding with Separate Image/Text Embeddings
# -----------------------------------------------------------
class CrossModalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Image Encoder: ResNet18 with a modified final layer
        self.image_encoder = models.resnet18(pretrained=True)
        self.image_encoder.fc = nn.Sequential(
            nn.Linear(512, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim)
        )

        # Text Encoder: BERT with an additional projection layer
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Sequential(
            nn.Linear(768, config.embedding_dim),  # BERT output dimension is 768
            nn.LayerNorm(config.embedding_dim)
        )

        # Learnable logit scale parameter for contrastive similarity scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        # Returns image embedding of shape (B, embedding_dim)
        return self.image_encoder(image)

    def encode_text(self, input_ids, attention_mask):
        # Gets [CLS] token representation from BERT and projects it to embedding_dim
        text_output = self.text_encoder(input_ids, attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]  # (B, 768)
        return self.text_projection(text_features)              # (B, embedding_dim)

    def forward(self, image, input_ids, attention_mask):
        # Obtain separate embeddings for image and text
        image_features = self.encode_image(image)                  # (B, D)
        text_features = self.encode_text(input_ids, attention_mask)  # (B, D)

        # Normalize embeddings for cosine similarity computation
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute the cosine similarity matrix scaled by learnable parameter
        logits = self.logit_scale.exp() * (image_features @ text_features.T)
        return logits


# -----------------------------------------------------------
# 3. Contrastive Loss Function
# -----------------------------------------------------------
def contrastive_loss(logits):
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    loss_img2txt = nn.CrossEntropyLoss()(logits, labels)
    loss_txt2img = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_img2txt + loss_txt2img) / 2


# -----------------------------------------------------------
# 4. Training Function
# -----------------------------------------------------------
def train_crossmodal_embedding(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        running_loss = 0
        num_batches = len(train_loader)

        for batch_idx ,batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, input_ids, attention_mask = [item.to(device) for item in batch]

            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask)
            loss = contrastive_loss(logits)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            running_loss += loss.item()

            # 100 배치마다 중간 손실 보고
            # if batch_idx % 100 == 0:
            #     print(f"Batch {batch_idx}/{num_batches}, Loss: {running_loss/100:.4f}")
            #     running_loss = 0

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")

        # Validation Phase
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


# -----------------------------------------------------------
# 5. Utility Function: Image Display
# -----------------------------------------------------------
def display_image(image_tensor):
    """
    Display an image tensor (C, H, W) using matplotlib.
    """
    img = image_tensor.cpu().clone()
    # If the image is normalized, unnormalize it
    if img.ndim == 3 and img.shape[0] == 3 and (img.min() < 0 or img.max() > 1):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
    img = torch.clamp(img, 0, 1)
    pil_img = to_pil_image(img)
    plt.imshow(pil_img)
    plt.axis('off')
    plt.show()


# -----------------------------------------------------------
# 6. Example Function for Demonstration
# -----------------------------------------------------------
def generate_example(model_path, image_dir, caption_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = namedtuple('Config', ['embedding_dim', 'image_dim', 'text_dim', 'hidden_dim', 'num_heads'])(
        embedding_dim=512, image_dim=512, text_dim=768, hidden_dim=512, num_heads=8
    )
    model = CrossModalEmbedding(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = Flickr8kDataset(image_dir, caption_file, transform=transform)
    _, val_dataset = torch.utils.data.random_split(
        dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        images, input_ids, attention_mask = next(iter(val_loader))
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = model(images, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        probs_t2i = torch.softmax(logits.T, dim=1)

        image_idx = 0
        topk_values, topk_indices = torch.topk(probs[image_idx], 5)
        print(f"Image {image_idx}:")
        display_image(images[image_idx])

        n_top = 3
        print(f"\nTop {n_top} Captions (Image -> Text):")
        for i in range(n_top):
            caption_idx = topk_indices[i]
            caption = dataset.tokenizer.decode(
                input_ids[caption_idx].cpu().numpy(), skip_special_tokens=True
            )
            print(f"  - {caption} (prob: {topk_values[i]:.4f})")

        caption_idx = 0
        topk_values_t2i, topk_indices_t2i = torch.topk(probs_t2i[caption_idx], 5)
        caption = dataset.tokenizer.decode(
            input_ids[caption_idx].cpu().numpy(), skip_special_tokens=True
        )
        print(f"\nCaption: {caption}")
        print(f"\nTop {n_top} Images (Text -> Image):")
        for i in range(n_top):
            image_idx = topk_indices_t2i[i]
            print(f" - Image {image_idx} (prob: {topk_values_t2i[i]:.4f})")
            display_image(images[image_idx])


# -----------------------------------------------------------
# 7. Main Execution Block
# -----------------------------------------------------------
if __name__ == '__main__':
    # Configuration for the model
    config = namedtuple('Config', ['embedding_dim', 'image_dim', 'text_dim', 'hidden_dim', 'num_heads'])(
        embedding_dim=512, image_dim=512, text_dim=768, hidden_dim=512, num_heads=8
    )

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set dataset paths (update these paths according to your environment)
    image_dir = './data/flickr8k/Images'
    caption_file = './data/flickr8k/captions.txt'

    dataset = Flickr8kDataset(image_dir, caption_file, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize and train the model
    model = CrossModalEmbedding(config)
    train_crossmodal_embedding(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4)

    # Save the trained model state
    torch.save(model.state_dict(), 'crossmodal_embedding_model.pth')
