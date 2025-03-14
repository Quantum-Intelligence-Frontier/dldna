# test_mistral_attention_mask.py
import torch
import pytest
from dldna.chapter_09.mistral.simple_mistral  import MistralConfig, MistralAttention

@pytest.fixture
def config():
    return MistralConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        sliding_window=4,
        use_cache=False
    )

def test_attention_mask_no_sliding_window(config):
    """Test attention mask without sliding window."""
    config.sliding_window = None  # Disable sliding window
    attention = MistralAttention(config)

    batch_size = 2
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # 1. No mask provided
    attn_output, _, _ = attention(hidden_states)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size)

    # 2. Full attention mask (no masking)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attn_output, _, _ = attention(hidden_states, attention_mask=attention_mask)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size)

    # 3. Partial attention mask (some positions masked)
    attention_mask = torch.tril(torch.ones(batch_size, seq_len, dtype=torch.bool))
    attn_output, _, _ = attention(hidden_states, attention_mask=attention_mask)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size)

    # 4. Full masking (all positions masked).  Should still work
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    attn_output, _, _ = attention(hidden_states, attention_mask=attention_mask)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size)

def test_attention_mask_with_sliding_window(config):
    """Test attention mask with sliding window."""
    attention = MistralAttention(config)

    batch_size = 2
    seq_len = 8  # Use a sequence length larger than the sliding window
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # 1. No additional mask provided, only sliding window
    attn_output, _, _ = attention(hidden_states)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size)

    # 2. Sliding window + full attention (no additional masking)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attn_output, _, _ = attention(hidden_states, attention_mask=attention_mask)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size)

    # 3. Sliding window + partial mask
    attention_mask = torch.tril(torch.ones(batch_size, seq_len, dtype=torch.bool))
    attn_output, _, _ = attention(hidden_states, attention_mask=attention_mask)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size)
    
    # 4. Full masking (all positions masked).  Should still work
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    attn_output, _, _ = attention(hidden_states, attention_mask=attention_mask)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size)


def test_attention_mask_different_dtypes(config):
    """Test attention masks with different dtypes."""
    config.sliding_window = None
    attention = MistralAttention(config)

    batch_size = 2
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Test different dtypes
    for dtype in [torch.bool, torch.int32, torch.float32, torch.float16]:
        if dtype == torch.bool:
          attention_mask = torch.ones(batch_size, seq_len, dtype=dtype)
        else:
          attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int32).to(dtype) #Make sure it is originally int
        
        attn_output, _, _ = attention(hidden_states, attention_mask=attention_mask)
        assert attn_output.shape == (batch_size, seq_len, config.hidden_size)
