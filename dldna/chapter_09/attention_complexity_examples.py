import numpy as np

def sliding_window_attention(q, k, v, window_size):
    """Sliding window attention
    Each token attends only to tokens within the surrounding window_size.
    
    Args:
        q, k, v: Query, key, and value tensors (batch_size, seq_len, dim)
        window_size: The range of surrounding tokens each token will attend to.
    """
    batch_size, seq_len, dim = q.shape
    attention_weights = np.zeros((batch_size, seq_len, seq_len))
    
    for i in range(seq_len):
        # Calculate the window range centered around the current position i
        start = max(0, i - window_size // 2)  # Window start point
        end = min(seq_len, i + window_size // 2 + 1)  # Window end point
        
        # Calculate attention scores between the current token and tokens within the window
        scores = np.matmul(q[:, i:i+1], k[:, start:end].transpose(0, 2, 1))
        # Normalize with softmax to generate weights
        attention_weights[:, i, start:end] = softmax(scores.squeeze(1), axis=-1)
    
    # Generate the final context vectors
    return np.matmul(attention_weights, v)

def sparse_block_attention(q, k, v, block_size):
    """Block sparse attention
    Divide the sequence into blocks and calculate attention within blocks.
    
    Args:
        q, k, v: Query, key, and value tensors (batch_size, seq_len, dim)
        block_size: The size of each block.
    """
    batch_size, seq_len, dim = q.shape
    num_blocks = seq_len // block_size  # Number of blocks
    attention_weights = np.zeros((batch_size, seq_len, seq_len))
    
    for i in range(num_blocks):
        # Range of the query block
        start_q = i * block_size
        end_q = (i + 1) * block_size
        
        for j in range(num_blocks):
            # Range of the key/value block
            start_k = j * block_size
            end_k = (j + 1) * block_size
            
            # Calculate attention scores between blocks
            scores = np.matmul(
                q[:, start_q:end_q], 
                k[:, start_k:end_k].transpose(0, 2, 1)
            )
            # Store attention weights block by block
            attention_weights[:, start_q:end_q, start_k:end_k] = softmax(scores, axis=-1)
    
    # Generate the final context vectors
    return np.matmul(attention_weights, v)

def low_rank_attention(q, k, v, rank):
    """Low-rank attention
    Project Q, K to a lower dimension to reduce computational complexity.
    
    Args:
        q, k, v: Query, key, and value tensors (batch_size, seq_len, dim)
        rank: The size of the lower dimension to project to.
    """
    batch_size, seq_len, dim = q.shape
    
    # Create random matrices for low-rank projection
    projection_q = np.random.randn(dim, rank) / np.sqrt(rank)  # Scaling
    projection_k = np.random.randn(dim, rank) / np.sqrt(rank)
    
    # Project Q, K to the lower dimension
    q_low = np.matmul(q, projection_q)  # (batch_size, seq_len, rank)
    k_low = np.matmul(k, projection_k)
    
    # Calculate attention in the lower dimension
    attention = np.matmul(q_low, k_low.transpose(0, 2, 1))
    attention_weights = softmax(attention, axis=-1)
    
    # Generate the final context vectors
    return np.matmul(attention_weights, v)

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Memory usage comparison
def calculate_memory(seq_len, method, window_size, block_size, rank):
    if method == "full":
        return seq_len * seq_len
    elif method == "sliding":
        return seq_len * window_size
    elif method == "sparse":
        return (seq_len // block_size) * (seq_len // block_size) * block_size * block_size
    elif method == "low_rank":
        return seq_len * rank * 2

def calcualte_efficieny():

    # Example data for testing
    batch_size = 2
    seq_len = 8
    dim = 4
    
    # Generate random Q, K, V matrices
    np.random.seed(42)
    q = np.random.randn(batch_size, seq_len, dim)
    k = np.random.randn(batch_size, seq_len, dim)
    v = np.random.randn(batch_size, seq_len, dim)
    
    print("Original input shape:", q.shape)
    
    # 1. Sliding Window Attention
    window_size = 4
    sliding_output = sliding_window_attention(q, k, v, window_size)
    print("\n1. Sliding Window Attention")
    print("Output shape:", sliding_output.shape)
    print("Output of the first batch, first token:", sliding_output[0, 0])

    # 2. Block Sparse Attention
    block_size = 2
    sparse_output = sparse_block_attention(q, k, v, block_size)
    print("\n2. Block Sparse Attention")
    print("Output shape:", sparse_output.shape)
    print("Output of the first batch, first token:", sparse_output[0, 0])

    # 3. Low-Rank Attention
    rank = 2
    low_rank_output = low_rank_attention(q, k, v, rank)
    print("\n3. Low-Rank Attention")
    print("Output shape:", low_rank_output.shape)
    print("Output of the first batch, first token:", low_rank_output[0, 0])


    print("\nMemory Usage Comparison (Relative Size):")
    print(f"Full Attention: {calculate_memory(seq_len, 'full', window_size, block_size, rank)}")
    print(f"Sliding Window: {calculate_memory(seq_len, 'sliding', window_size, block_size, rank)}")
    print(f"Block Sparse: {calculate_memory(seq_len, 'sparse', window_size, block_size, rank)}")
    print(f"Low Rank: {calculate_memory(seq_len, 'low_rank', window_size, block_size, rank)}")


if __name__ == "__main__":
    calcualte_efficieny()