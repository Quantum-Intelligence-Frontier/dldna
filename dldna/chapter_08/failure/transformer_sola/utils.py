import torch

def create_attention_mask(input_ids, pad_token_id):
    """Create attention mask for input sequence."""
    return (input_ids != pad_token_id).float().unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_length):
    """Create causal mask for self-attention in decoder."""
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)