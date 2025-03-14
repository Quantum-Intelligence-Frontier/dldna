import numpy as np
import torch

def visualize_position_embedding():
    # Example word embeddings
    word_embeddings = {
        'I': np.array([0.2, 0.3, 0.1, 0.4]),  # 4-dimensional embedding vector
        'love': np.array([0.5, 0.2, 0.8, 0.1]),
        'deep': np.array([0.3, 0.7, 0.2, 0.5]),
        'learning': np.array([0.6, 0.4, 0.3, 0.2])
    }

    # Word order list
    words = ['I', 'love', 'deep', 'learning']

    # Convert embedding vectors into a matrix
    embeddings = np.vstack([word_embeddings[word] for word in words])

    print("1. Original embedding matrix:")
    print_matrix_with_labels(embeddings, words, [f"dim{i+1}" for i in range(embeddings.shape[1])])
    print("\nEach row is the embedding vector of a word")
    print("-" * 50)

    # Create position indices
    position_indices = np.arange(len(words))
    print("\n2. Position indices:")
    print(position_indices)  # Print in word order
    print("\nIndices representing the position of each word (starting from 0)")
    print("-" * 50)

    # Add position information (simple addition)
    position_added_embeddings = embeddings + position_indices.reshape(-1, 1) # Broadcasting
    print("\n3. Embeddings with position information added:")
    print_matrix_with_labels(position_added_embeddings, words, [f"dim{i+1}" for i in range(embeddings.shape[1])])
    print("\nResult of adding position indices to each embedding vector (broadcasting)")
    print("-" * 50)

    # Compare how position information changed
    print("\n4. Changes due to adding position information:")
    for i, word in enumerate(words):
        print(f"\n{word} ({position_indices[i]}):")  # Print word and position index
        print(f"  Original:     {embeddings[i]}")
        print(f"  Pos. Added: {position_added_embeddings[i]}")
        print(f"  Difference:     {position_added_embeddings[i] - embeddings[i]}")

def print_matrix_with_labels(matrix, row_labels, col_labels):
    """Prints a matrix with labels (supports both NumPy arrays and tensors)."""

    # Calculate maximum label length
    max_row_label_len = max(len(label) for label in row_labels)
    max_col_label_len = max(len(label) for label in col_labels)
    col_width = max(max_row_label_len, max_col_label_len) + 2

    # Print header
    header = " " * (col_width) + "".join([f"{label:>{col_width}}" for label in col_labels])
    print(header)

    # Print each row
    for i, row_label in enumerate(row_labels):
        row_str = f"{row_label:<{col_width}}" # Row label

        if isinstance(matrix, np.ndarray):
            row_values = [f"{matrix[i, j]:.2f}" for j in range(matrix.shape[1])]
        elif isinstance(matrix, torch.Tensor):
             row_values = [f"{matrix[i, j].item():.2f}" for j in range(matrix.size(1))]
        else:
             raise ValueError("Unsupported matrix type")

        row_str += "".join(f"[{value:>{col_width-2}}]" for value in row_values)
        print(row_str)
