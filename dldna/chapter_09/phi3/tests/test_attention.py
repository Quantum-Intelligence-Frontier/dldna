# test_phi_mini_attention.py
import torch
import pytest
from dldna.chapter_09.phi3.simple_phi3 import PhiMiniConfig, PhiMiniAttention  # Classes from the simple_phi3 file

def create_config():
    cfg = PhiMiniConfig(
        vocab_size=100,
        hidden_size=256,           # Use a small hidden_size
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,     # 256/8 = 32, calculate head_dim
        num_key_value_heads=8,
        max_position_embeddings=64,
        rope_theta=10000.0
    )
    # Add a default value to control cache usage for testing
    cfg.use_cache = False
    return cfg

def test_phi_mini_attention_forward():
    # Test basic forward operation (explicitly passing use_cache=True)
    config = create_config()
    attention_module = PhiMiniAttention(config)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    attn_output, attn_weights, past_key_values = attention_module(x, use_cache=True)
    # The shape of the output tensor should be (B, T, hidden_size)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected {(batch_size, seq_len, config.hidden_size)} but got {attn_output.shape}"
    # past_key_values should be a tuple
    assert isinstance(past_key_values, tuple), "past_key_values should be a tuple"
    assert len(past_key_values) == 2, "past_key_values should contain two elements (key_states and value_states)"

def test_phi_mini_attention_with_attention_mask():
    # Test forward operation when attention_mask option is provided (use_cache=False)
    config = create_config()
    attention_module = PhiMiniAttention(config)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    # Typically, the shape of attention_mask is (B, 1, T, T)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    attn_output, attn_weights, past_key_values = attention_module(x, attention_mask=attention_mask)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected {(batch_size, seq_len, config.hidden_size)} but got {attn_output.shape}"


def test_phi_mini_attention_invalid_input():
    # Test if an error occurs when the input tensor has an incorrect dimension
    config = create_config()
    attention_module = PhiMiniAttention(config)
    # Pass a 2D tensor without the last feature dimension
    x_invalid = torch.randn(2, 10)
    with pytest.raises(Exception):
        attention_module(x_invalid)

def test_phi_mini_attention_backward():
    # Check if backpropagation is performed correctly and gradients are generated for each parameter
    config = create_config()
    attention_module = PhiMiniAttention(config)
    batch_size, seq_len = 2, 10
    # Enable requires_grad for the input tensor
    x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
    attn_output, _, _ = attention_module(x)
    loss = attn_output.sum()
    loss.backward()
    for name, param in attention_module.named_parameters():
        assert param.grad is not None, f"Gradient for {name} was not computed"

def test_phi_mini_attention_with_attention_mask():
    # Test if the forward operation is performed correctly even when the attention_mask option is provided
    config = create_config()
    attention_module = PhiMiniAttention(config)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    # Usually, the shape of the attention mask is provided as (B, 1, seq_len, seq_len)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
    attn_output, attn_weights, past_key_values = attention_module(x, attention_mask=attention_mask)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected {(batch_size, seq_len, config.hidden_size)} but got {attn_output.shape}"

def test_phi_mini_attention_eval_mode():
    # Test if the model maintains the same output shape even in evaluation mode
    config = create_config()
    attention_module = PhiMiniAttention(config)
    attention_module.eval()
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    with torch.no_grad():
        attn_output, attn_weights, past_key_values = attention_module(x)
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected {(batch_size, seq_len, config.hidden_size)} but got {attn_output.shape}"

# ------------------ Additional Test Cases ------------------

def test_phi_mini_attention_output_attentions():
    # Test if attention weights are returned when called with output_attentions=True
    config = create_config()
    attention_module = PhiMiniAttention(config)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    attn_output, attn_weights, past_key_values = attention_module(x, output_attentions=True)
    assert attn_weights is not None, "When output_attentions is True, attention_weights should not be None"
    expected_shape = (batch_size, config.num_attention_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_shape, \
        f"Expected attention weights shape {expected_shape} but got {attn_weights.shape}"

def test_phi_mini_attention_attention_weights_sum():
    # Check if the softmax result of attention weights sums to 1 along the last dimension
    config = create_config()
    attention_module = PhiMiniAttention(config)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    _, attn_weights, _ = attention_module(x, output_attentions=True)
    # The sum along the last dimension (attention distribution for each query) should be 1
    attn_sum = attn_weights.sum(dim=-1)
    ones = torch.ones_like(attn_sum)
    assert torch.allclose(attn_sum, ones, atol=1e-5), "Attention weights do not sum to 1 along the last dimension"

def test_phi_mini_attention_use_cache():
    # Test if past_key_values are returned when called with use_cache=True
    config = create_config()
    # Explicitly enable cache usage: set config.use_cache to True
    config.use_cache = True
    attention_module = PhiMiniAttention(config)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    _, _, past_key_values = attention_module(x, use_cache=True)
    assert past_key_values is not None, "With use_cache=True, past_key_values should not be None"
    assert isinstance(past_key_values, tuple) and len(past_key_values) == 2, \
        "past_key_values must be a tuple of (key_states, value_states)"

def test_phi_mini_attention_invalid_attention_mask():
    # Test if an error occurs when an attention_mask with an incorrect shape is input
    config = create_config()
    attention_module = PhiMiniAttention(config)
    batch_size, seq_len = 2, 10
    # The expected shape is (B, 1, seq_len, seq_len) or (B, seq_len, seq_len)
    # Here, intentionally pass a shape mismatch (B, seq_len, seq_len+1)
    invalid_attention_mask = torch.ones(batch_size, seq_len, seq_len+1)
    with pytest.raises(Exception):
        attention_module(torch.randn(batch_size, seq_len, config.hidden_size), attention_mask=invalid_attention_mask)

def test_phi_mini_attention_past_key_value():
    # Test incremental decoding with past_key_value
    config = create_config()
    config.use_cache = True  # Ensure cache is enabled
    attention_module = PhiMiniAttention(config)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # First, run a forward pass without cache to get initial past_key_value
    _, _, past_key_value = attention_module(x, use_cache=True)
    assert past_key_value is not None
    assert len(past_key_value) == 2
    assert past_key_value[0].shape[0] == batch_size  # Check batch size
    assert past_key_value[1].shape[0] == batch_size
    assert past_key_value[0].shape[1] == config.num_attention_heads  # Check num heads
    assert past_key_value[1].shape[1] == config.num_attention_heads
    assert past_key_value[0].shape[2] == seq_len # Check sequence len (should be same as initial)
    assert past_key_value[1].shape[2] == seq_len
    assert past_key_value[0].shape[3] == config.hidden_size // config.num_attention_heads  # Check head dim
    assert past_key_value[1].shape[3] == config.hidden_size // config.num_attention_heads

    # Then, run a second forward pass WITH past_key_value, simulating a single new token
    x_new = torch.randn(batch_size, 1, config.hidden_size)  # Single new token
    attn_output_new, _, past_key_value_new = attention_module(x_new, use_cache=True, past_key_value=past_key_value)

    # Output shape should be (batch_size, 1, config.hidden_size) for the single new token
    assert attn_output_new.shape == (batch_size, 1, config.hidden_size)

    # Check that past_key_value_new is updated correctly
    assert past_key_value_new is not None
    assert len(past_key_value_new) == 2
    # Sequence length should have increased by 1 (new token added)
    assert past_key_value_new[0].shape[2] == seq_len + 1
    assert past_key_value_new[1].shape[2] == seq_len + 1
    # Other dimensions should remain the same
    assert past_key_value_new[0].shape[0] == batch_size
    assert past_key_value_new[1].shape[0] == batch_size
    assert past_key_value_new[0].shape[1] == config.num_attention_heads
    assert past_key_value_new[1].shape[1] == config.num_attention_heads    
    assert past_key_value_new[0].shape[3] == config.hidden_size // config.num_attention_heads
    assert past_key_value_new[1].shape[3] == config.hidden_size // config.num_attention_heads


    # Test with a different sequence length for the new token (should still work)
    x_new_2 = torch.randn(batch_size, 5, config.hidden_size) # 5 new tokens
    attn_output_new_2, _, past_key_value_new_2 = attention_module(x_new_2, use_cache=True, past_key_value=past_key_value)
    assert attn_output_new_2.shape == (batch_size, 5, config.hidden_size)
    assert past_key_value_new_2[0].shape[2] == seq_len + 5  # Now seq_len + 5
    assert past_key_value_new_2[1].shape[2] == seq_len + 5
    
    # Test with attention mask and past_key_values
    attention_mask = torch.ones(batch_size, 1, 1, seq_len + 5) # Corrected mask shape
    attn_output_new_3, _, past_key_value_new_3 = attention_module(x_new_2, use_cache=True, past_key_value=past_key_value, attention_mask=attention_mask)
    assert attn_output_new_3.shape == (batch_size, 5, config.hidden_size)    
    assert past_key_value_new_3[0].shape[2] == seq_len + 5  # Now seq_len + 5
    assert past_key_value_new_3[1].shape[2] == seq_len + 5


def test_phi_mini_attention_zero_length_initial():
    config = create_config()
    config.use_cache = True
    attention_module = PhiMiniAttention(config)
    batch_size = 2
    # Zero-length initial sequence
    x = torch.randn(batch_size, 0, config.hidden_size)
    attn_output, _, past_key_value = attention_module(x, use_cache=True)
    assert attn_output.shape == (batch_size, 0, config.hidden_size)
    # Even with zero-length, past_key_value should be properly initialized
    assert past_key_value is not None
    assert len(past_key_value) == 2
    assert past_key_value[0].shape[2] == 0
    assert past_key_value[1].shape[2] == 0

    # Follow up with a single token input
    x_new = torch.randn(batch_size, 1, config.hidden_size)
    attn_output_new, _, past_key_value_new = attention_module(x_new, use_cache=True, past_key_value=past_key_value)
    assert attn_output_new.shape == (batch_size, 1, config.hidden_size)
    assert past_key_value_new[0].shape[2] == 1
    assert past_key_value_new[1].shape[2] == 1

def test_phi_mini_attention_single_token_initial():
    config = create_config()
    config.use_cache = True
    attention_module = PhiMiniAttention(config)
    batch_size = 2
    # Single-token initial sequence
    x = torch.randn(batch_size, 1, config.hidden_size)
    attn_output, _, past_key_value = attention_module(x, use_cache=True)
    assert attn_output.shape == (batch_size, 1, config.hidden_size)
    assert past_key_value is not None
    assert len(past_key_value) == 2
    assert past_key_value[0].shape[2] == 1
    assert past_key_value[1].shape[2] == 1

    # Follow up with another single token input
    x_new = torch.randn(batch_size, 1, config.hidden_size)
    attn_output_new, _, past_key_value_new = attention_module(x_new, use_cache=True, past_key_value=past_key_value)
    assert attn_output_new.shape == (batch_size, 1, config.hidden_size)
    assert past_key_value_new[0].shape[2] == 2
    assert past_key_value_new[1].shape[2] == 2