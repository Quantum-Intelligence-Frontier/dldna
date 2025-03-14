import torch

def create_pad_mask(seq, pad_idx=0):
    """Create padding mask"""
    # Create in the shape [batch_size, 1, 1, seq_len]
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_subsequent_mask(size):
    """Create subsequent mask for the decoder"""
    # Create in the shape [1, size, size]
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0