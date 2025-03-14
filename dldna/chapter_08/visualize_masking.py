import torch
import torch.nn.functional as F
import numpy as np

def visualize_causal_mask():
    # Example sentence and word vectors
    words = ["I", "love", "deep", "learning"]
    # (4,4) scores
    scores = torch.tensor([
        [0.9, 0.7, 0.3, 0.2],  # I -> I, love, deep, learning
        [0.6, 0.8, 0.9, 0.4],  # love -> I, love, deep, learning
        [0.2, 0.5, 0.7, 0.9],  # deep -> I, love, deep, learning
        [0.4, 0.3, 0.8, 0.6]   # learning -> I, love, deep, learning
    ])

    print("1. Original attention score matrix:")
    print_matrix_with_labels(scores, words, words)
    print("\nEach row represents the attention scores from the current position to all positions")
    print("-" * 50)

    # 2. Create mask
    seq_len = len(words)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    print("\n2. Lower triangular mask (1: allowed, 0: blocked):")
    print_matrix_with_labels(mask, words, words)
    print("\nOnly the diagonal and below are 1, the rest are 0")
    print("-" * 50)

    # 3. Masked matrix with -inf
    inf_mask = mask.masked_fill(mask == 0, float('-inf'))
    print("\n3. Mask converted to -inf:")
    print_matrix_with_labels(inf_mask, words, words, fmt=".1e") # exponential notation
    print("\nConverting 0 to -inf so that it becomes 0 after softmax")
    print("-" * 50)

    # 4. Masked scores
    masked_scores = scores + inf_mask
    print("\n4. Attention scores with mask applied:")
    print_matrix_with_labels(masked_scores, words, words, fmt=".1f")
    print("\nFuture information (upper triangle) is masked with -inf")
    print("-" * 50)

    # 5. Result after softmax
    weights = F.softmax(masked_scores, dim=-1)
    print("\n5. Final attention weights (after softmax):")
    print_matrix_with_labels(weights, words, words)
    print("\nThe sum of each row becomes 1, and future information is masked to 0")

def visualize_padding_mask():
   # Example sentences (padded sequences)
    words = ["I", "love", "cats", "PAD"]
    sequences = torch.tensor([
      [1, 1, 1, 0],  # I love cats PAD
      [1, 1, 0, 0],  # I love PAD PAD
      [1, 1, 1, 1],  # I love cats dogs
      [1, 0, 0, 0]   # I PAD PAD PAD
     ])

    print("1. Input sequences (1: actual token, 0: padding):")
    print(sequences)
    print("\nEach row represents sentences of different lengths, padded to the same length")
    print("-" * 50)

    # 2. Create padding mask
    padding_mask = (sequences != 0).float().unsqueeze(1)  # (batch_size, 1, seq_len)
    print("\n2. Create padding mask (1: valid token, 0: padding token):")
    print(padding_mask) # (batch_size, 1, seq_len)
    print("\nPositions that are not padding (0) are 1, padding positions are 0")
    print("-" * 50)


    # 3. Example attention scores (the weight values are not important here, as the purpose is to visualize the padding mask)
    attention_scores = torch.tensor([
      [0.9, 0.7, 0.3, 0.2],
      [0.6, 0.8, 0.9, 0.4],
      [0.2, 0.5, 0.7, 0.9],
      [0.4, 0.3, 0.8, 0.6]
    ]).unsqueeze(0).repeat(4, 1, 1) # (batch_size, seq_len, seq_len)

    print("\n3. Original attention scores (first sentence):")
    print_matrix_with_labels(attention_scores[0], words, words)
    print("\nAttention scores at each position")
    print("-" * 50)

    # 4. Apply padding mask
    masked_scores = attention_scores.masked_fill(padding_mask == 0, float('-inf'))
    print("\n4. Scores with padding mask applied (first sentence):")
    print_matrix_with_labels(masked_scores[0], words, words, fmt=".1e") # Use .1e to display -inf
    print("\nThe scores at padding positions are masked with -inf")
    print("-" * 50)

    # 5. Apply softmax
    attention_weights = F.softmax(masked_scores, dim=-1)
    print("\n5. Final attention weights (first sentence):")
    print_matrix_with_labels(attention_weights[0], words, words)
    print("\nThe weights at padding positions become 0, and the sum of the weights at the remaining positions is 1")

def print_matrix_with_labels(matrix, row_labels, col_labels, fmt=".2f"):
    """Prints a matrix with labels."""

    # Calculate maximum label length
    max_row_label_len = max(len(label) for label in row_labels)
    max_col_label_len = max(len(label) for label in col_labels)
    col_width = max(max_row_label_len, max_col_label_len) + 4 # Adjusted to max label length

    # Print header
    header = " " * (col_width) + "".join([f"{label:>{col_width}}" for label in col_labels])
    print(header)

    # Print each row
    for i, row_label in enumerate(row_labels):
        row_str = f"{row_label:<{col_width}}" # Row label
        row_values = [f"{matrix[i, j]:{fmt}}" for j in range(matrix.size(1))]
        row_str += "".join(f"[{value:>{col_width-2}}]" for value in row_values)
        print(row_str)

