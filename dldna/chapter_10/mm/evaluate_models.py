# from PIL import Image
# import torch
# from torchvision import transforms
# from transformers import BertTokenizer
# import os
# import pandas as pd
# # Assuming dldna.chapter_10.mm.train_multimodal is in the same directory or Python path.
# # Adjust as necessary (e.g., from train_multimodal import ...).
# from dldna.chapter_10.mm.train_multimodal import ImageTextMatchingModel, get_cross_attention

# def evaluate_model(model_path, image_path, test_captions, config):
#     """
#     Loads a saved model, calculates similarities between a given image and captions,
#     and returns the caption with the highest similarity.

#     Args:
#         model_path (str): Path to the saved model file.
#         image_path (str): Path to the test image file.
#         test_captions (list): List of test captions.
#         config (dict): Model configuration.

#     Returns:
#         tuple: (Caption with highest similarity, highest similarity score, list of similarities for all captions)
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load the model
#     model = ImageTextMatchingModel()  # Model structure definition (refer to train_multimodal.py)

#     # Load the Cross Attention model
#     model_version = model_path.split('_')[-1].split('.')[0]  # 'model_final_v2.pth' -> 'v2'
#     if model_version.startswith('v'):
#         model.cross_attention = get_cross_attention(model_version, config=config)
#     else:
#         model_version = 'v0'
#         model.cross_attention = get_cross_attention(model_version, config=config)


#     # Load state_dict, handling potential missing keys
#     state_dict = torch.load(model_path, map_location=device)

#     # Check for and remove 'cross_attention.norm.weight' and 'cross_attention.norm.bias' keys,
#     # if they exist (these keys are not present in v0, v1, v2, but are present in v3 onwards).
#     if 'cross_attention.norm.weight' in state_dict:
#         del state_dict['cross_attention.norm.weight']
#     if 'cross_attention.norm.bias' in state_dict:
#         del state_dict['cross_attention.norm.bias']

#     model.load_state_dict(state_dict, strict=False)  # Load weights
#     model.to(device)
#     model.eval()

#     # Image preprocessing
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     # Text tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     similarities = []
#     with torch.no_grad():
#         for caption in test_captions:
#             tokens = tokenizer(caption, padding='max_length', truncation=True,
#                               max_length=64, return_tensors="pt")
#             input_ids = tokens['input_ids'].to(device)
#             attention_mask = tokens['attention_mask'].to(device)

#             logits = model(image_tensor, input_ids, attention_mask)
#             similarity = logits[0, 0].item()
#             similarities.append(similarity)

#     # Find the caption with the highest similarity
#     max_similarity = max(similarities)
#     max_index = similarities.index(max_similarity)
#     best_caption = test_captions[max_index]

#     return best_caption, max_similarity, similarities


# def evaluate_all_models(model_dir, image_path, test_captions, model_versions):
#     """
#     Evaluates models in a given directory and returns results as a DataFrame.

#     Args:
#       model_dir: Directory containing the model files
#       image_path: Path to the test image
#       test_captions: List of test captions
#       model_versions: List of model versions to evaluate

#     Returns:
#       pd.DataFrame
#     """

#     results = []
#     config = {'dim': 256}

#     for model_version in model_versions:  # Use the specified list of versions
#         model_file = f"model_final_{model_version}.pth"  # Construct the filename
#         model_path = os.path.join(model_dir, model_file)

#         # Check if the file exists:
#         if not os.path.exists(model_path):
#             print(f"Warning: Model file not found: {model_path}. Skipping.")
#             continue


#         best_caption, max_similarity, similarities = evaluate_model(model_path, image_path, test_captions, config)

#         is_trained_well = best_caption == "A cat sleeping on a couch"

#         similarities_copy = similarities.copy()
#         similarities_copy.remove(max_similarity)
#         second_max_similarity = max(similarities_copy) if similarities_copy else -float('inf')

#         similarity_ratio = max_similarity / (second_max_similarity + 1e-9) if second_max_similarity != -float('inf') else float('inf')
#         similarity_gap = max_similarity - second_max_similarity

#         results.append({
#             'model_version': model_version,
#             # 'model_path': model_path,  # Remove model_path column
#             'best_caption': best_caption,
#             'max_similarity': f"{max_similarity:.3f}",
#             '2nd_max_similarity': f"{second_max_similarity:.3f}",
#             'similarity_ratio': f"{similarity_ratio:.3f}",
#             'similarity_gap': f"{similarity_gap:.3f}",
#             'trained_well': is_trained_well,
#             'all_similarities': [f"{sim:.3f}" for sim in similarities]
#         })

#     df_results = pd.DataFrame(results)
#     df_results['similarity_ratio_rank'] = df_results['similarity_ratio'].astype(float).rank(ascending=False, method='min').astype(int)
#     df_results = df_results[['model_version', 'best_caption', 'all_similarities',
#                              'similarity_ratio', 'similarity_gap', 'trained_well',
#                              'similarity_ratio_rank']] # Specify column order

#     return df_results



# # Test captions (fixed)
# test_captions = [
#     "A dog playing in the park",
#     "A cat sleeping on a couch",
#     "Children playing soccer",
#     "A sunset over the ocean",
#     "A person cooking in the kitchen"
# ]

# # Run model evaluation
# image_path = './cat_resized.png'
# model_dir = '.'
# model_versions = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10_1', 'v10_2', 'v10_3', 'v10_4', 'v10_5', 'v10_6', 'v11']

# results_df = evaluate_all_models(model_dir, image_path, test_captions, model_versions)

# # Print results (Markdown table)
# print(results_df.to_markdown(index=False))

# # Print results (detailed)
# for _, row in results_df.iterrows():
#     print(f"\nModel: {row['model_version']}")
#     print(f"  Best Caption: {row['best_caption']}")
#     print(f"  Trained Well: {row['trained_well']}")
#     print(f"  Similarity Ratio: {row['similarity_ratio']}")
#     print(f"  Similarity Gap: {row['similarity_gap']}")
#     print("  All Similarities:")
#     for caption, sim in zip(test_captions, row['all_similarities']):
#         print(f"    - {caption:<30}: {sim}")


from PIL import Image
import torch
from torchvision import transforms
from transformers import BertTokenizer
import os
import pandas as pd
# Assuming dldna.chapter_10.mm.train_multimodal is in the same directory or Python path.
# Adjust as necessary (e.g., from train_multimodal import ...).
from dldna.chapter_10.mm.train_multimodal import ImageTextMatchingModel, get_cross_attention

def evaluate_model(model_path, image_path, test_captions, config):
    """
    Loads a saved model, calculates similarities between a given image and captions,
    and returns the caption with the highest similarity.

    Args:
        model_path (str): Path to the saved model file.
        image_path (str): Path to the test image file.
        test_captions (list): List of test captions.
        config (dict): Model configuration.

    Returns:
        tuple: (Caption with highest similarity, highest similarity score, list of similarities for all captions)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = ImageTextMatchingModel(image_encoder_dim=2048, text_encoder_dim=768, projection_dim=config['dim'])  # Model structure definition (refer to train_multimodal.py)

    # Load the Cross Attention model
    model_version = model_path.split('_')[-1].split('.')[0]  # 'model_final_v2.pth' -> 'v2'
    if model_version.startswith('v'):
        model.cross_attention = get_cross_attention(model_version, config=config)
    else:
        model_version = 'v0'
        model.cross_attention = get_cross_attention(model_version, config=config)


    # Load state_dict, handling potential missing keys
    state_dict = torch.load(model_path, map_location=device)

    # Check for and remove 'cross_attention.norm.weight' and 'cross_attention.norm.bias' keys,
    # if they exist (these keys are not present in v0, v1, v2, but are present in v3 onwards).
    # Also check for and remove other potentially problematic keys.
    keys_to_remove = [
        'cross_attention.norm.weight', 'cross_attention.norm.bias',
        'cross_attention.image_proj.weight', 'cross_attention.image_proj.bias',
        'cross_attention.text_proj.weight', 'cross_attention.text_proj.bias',
        'cross_attention.attn_norm.weight', 'cross_attention.attn_norm.bias',
        'cross_attention.ff_norm.weight', 'cross_attention.ff_norm.bias',
        'cross_attention.to_out.weight', 'cross_attention.to_out.bias',
        'cross_attention.out_norm.weight', 'cross_attention.out_norm.bias'
    ]
    for key in keys_to_remove:
      if key in state_dict:
          del state_dict[key]

    model.load_state_dict(state_dict, strict=False)  # Load weights
    model.to(device)
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Text tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    similarities = []
    with torch.no_grad():
        for caption in test_captions:
            tokens = tokenizer(caption, padding='max_length', truncation=True,
                              max_length=64, return_tensors="pt")
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            logits = model(image_tensor, input_ids, attention_mask)
            similarity = logits[0, 0].item()
            similarities.append(similarity)

    # Find the caption with the highest similarity
    max_similarity = max(similarities)
    max_index = similarities.index(max_similarity)
    best_caption = test_captions[max_index]

    return best_caption, max_similarity, similarities


def evaluate_all_models(model_dir, image_path, test_captions, model_versions):
    """
    Evaluates models in a given directory and returns results as a DataFrame.

    Args:
      model_dir: Directory containing the model files
      image_path: Path to the test image
      test_captions: List of test captions
      model_versions: List of model versions to evaluate

    Returns:
      pd.DataFrame
    """

    results = []
    config = {'dim': 256}  # Initialize config here

    for model_version in model_versions:  # Use the specified list of versions
        model_file = f"model_final_{model_version}.pth"  # Construct the filename
        model_path = os.path.join(model_dir, model_file)

        # Check if the file exists:
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}. Skipping.")
            continue

        # Pass config to evaluate_model
        best_caption, max_similarity, similarities = evaluate_model(model_path, image_path, test_captions, config)

        is_trained_well = best_caption == "A cat sleeping on a couch"

        similarities_copy = similarities.copy()
        similarities_copy.remove(max_similarity)
        second_max_similarity = max(similarities_copy) if similarities_copy else -float('inf')

        similarity_ratio = max_similarity / (second_max_similarity + 1e-9) if second_max_similarity != -float('inf') else float('inf')
        similarity_gap = max_similarity - second_max_similarity

        results.append({
            'model_version': model_version,
            # 'model_path': model_path,  # Remove model_path column
            'best_caption': best_caption,
            'max_similarity': f"{max_similarity:.3f}",
            '2nd_max_similarity': f"{second_max_similarity:.3f}",
            'similarity_ratio': f"{similarity_ratio:.3f}",
            'similarity_gap': f"{similarity_gap:.3f}",
            'trained_well': is_trained_well,
            'all_similarities': [f"{sim:.3f}" for sim in similarities]
        })

    df_results = pd.DataFrame(results)
    df_results['similarity_ratio_rank'] = df_results['similarity_ratio'].astype(float).rank(ascending=False, method='min').astype(int)
    df_results = df_results[['model_version', 'best_caption', 'all_similarities',
                             'similarity_ratio', 'similarity_gap', 'trained_well',
                             'similarity_ratio_rank']] # Specify column order

    return df_results



# Test captions (fixed)
test_captions = [
    "A dog playing in the park",
    "A cat sleeping on a couch",
    "Children playing soccer",
    "A sunset over the ocean",
    "A person cooking in the kitchen"
]

# Run model evaluation
image_path = './cat_resized.png'
model_dir = '.'
model_versions = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10_1', 'v10_2', 'v10_3', 'v10_4', 'v10_5', 'v10_6', 'v11']

results_df = evaluate_all_models(model_dir, image_path, test_captions, model_versions)

# Print results (Markdown table)
print(results_df.to_markdown(index=False))

# Print results (detailed)
for _, row in results_df.iterrows():
    print(f"\nModel: {row['model_version']}")
    print(f"  Best Caption: {row['best_caption']}")
    print(f"  Trained Well: {row['trained_well']}")
    print(f"  Similarity Ratio: {row['similarity_ratio']}")
    print(f"  Similarity Gap: {row['similarity_gap']}")
    print("  All Similarities:")
    for caption, sim in zip(test_captions, row['all_similarities']):
        print(f"    - {caption:<30}: {sim}")