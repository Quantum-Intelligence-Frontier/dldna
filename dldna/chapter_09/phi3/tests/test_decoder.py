# test_phi_mini_decoder_layer.py
import torch
import pytest
from dldna.chapter_09.phi3.simple_phi3 import PhiMiniConfig, PhiMiniDecoderLayer,PhiMiniForCausalLM


def create_config():
    config = PhiMiniConfig(
        vocab_size=100,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=64,
        rope_theta=10000.0
    )
    return config

def test_decoder_layer_with_attention_and_cache():
    # When both output_attentions and use_cache options are True
    # Check if the length of the returned tuple is 3 and the shape of each tensor is correct.
    config = create_config()
    decoder_layer = PhiMiniDecoderLayer(config, layer_idx=0)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    outputs = decoder_layer(
        x,
        attention_mask=torch.ones(batch_size, seq_len, seq_len),
        past_key_value=None,
        output_attentions=True,
        use_cache=True
    )
    # expected outputs: (hidden_states, attn_weights, past_key_value)
    assert len(outputs) == 3, "Output tuple length should be 3 when both output_attentions and use_cache are True"
    hidden_states, attn_weights, past_key_value = outputs
    assert hidden_states.shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected hidden_states shape {(batch_size, seq_len, config.hidden_size)} but got {hidden_states.shape}"
    expected_attn_shape = (batch_size, config.num_attention_heads, seq_len, seq_len)
    assert attn_weights is not None, "Attention weights should not be None when output_attentions is True"
    assert attn_weights.shape == expected_attn_shape, f"Expected attention weights shape {expected_attn_shape} but got {attn_weights.shape}"
    assert past_key_value is not None, "past_key_value should not be None when use_cache is True"
    # Briefly check only the batch dimension of key and value
    k, v = past_key_value
    assert k.shape[0] == batch_size and v.shape[0] == batch_size, "past_key_value tensors should have the correct batch size"

def test_decoder_layer_repeated_eval_consistency():
    # Check if the results are consistent when called repeatedly with the same input in evaluation mode.
    config = create_config()
    decoder_layer = PhiMiniDecoderLayer(config, layer_idx=0)
    decoder_layer.eval()
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    with torch.no_grad():
        out1 = decoder_layer(
            x,
            attention_mask=torch.ones(batch_size, seq_len, seq_len),
            past_key_value=None,
            output_attentions=True,
            use_cache=False
        )
        out2 = decoder_layer(
            x,
            attention_mask=torch.ones(batch_size, seq_len, seq_len),
            past_key_value=None,
            output_attentions=True,
            use_cache=False
        )
    # Compare hidden_states and attention weights (ignore cache as it's not returned)
    for t1, t2 in zip(out1[:2], out2[:2]):
        assert torch.allclose(t1, t2, atol=1e-5), "Repeated eval outputs should be identical in eval mode"

def test_decoder_layer_invalid_attention_mask_shape():
    # An exception should be raised if an attention mask with an incorrect shape is passed.
    config = create_config()
    decoder_layer = PhiMiniDecoderLayer(config, layer_idx=0)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    # The correct shape is (B, T, T), but here we pass (B, T+1, T)
    invalid_mask = torch.ones(batch_size, seq_len + 1, seq_len)
    with pytest.raises(Exception):
        decoder_layer(
            x,
            attention_mask=invalid_mask,
            past_key_value=None,
            output_attentions=False,
            use_cache=False
        )

def test_phi_mini_for_causal_lm_forward():
    # This is a test to verify the forward pass and loss calculation of PhiMiniForCausalLM.
    config = create_config()
    config.num_hidden_layers = 1  # Configure a single layer for a simple test
    model = PhiMiniForCausalLM(config)
    batch_size, seq_len = 2, 10
    # Input tokens (generate random numbers within the vocab range)
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len))
    # For training purposes, use the same sequence as the input sequence as labels (shifted)
    labels = input_ids.clone()
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True
    )
    logits = outputs["logits"]
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected logits shape {(batch_size, seq_len, config.vocab_size)} but got {logits.shape}"
    loss = outputs["loss"]
    assert loss is not None, "Loss should be computed when labels are provided"
    # Whether past_key_values and hidden_states are returned may depend on the model
    # If logits and loss are normal, training is possible.

def test_phi_mini_for_causal_lm_generate():
    # Check if the generate function of PhiMiniForCausalLM works without errors.
    config = create_config()
    config.num_hidden_layers = 1  # Configure simply for testing
    model = PhiMiniForCausalLM(config)
    # Simple input for a single batch
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    # Set pad_token_id and eos_token_id if necessary
    config.pad_token_id = 0
    config.eos_token_id = 2
    output_ids = model.generate(input_ids, max_new_tokens=5, do_sample=True)
    assert output_ids.shape[1] >= input_ids.shape[1], "Generated sequence length should be at least the input length"