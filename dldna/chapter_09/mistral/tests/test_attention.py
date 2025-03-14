import unittest
import torch
from dldna.chapter_09.mistral.simple_mistral import MistralAttention, MistralConfig, MistralRotaryEmbedding, apply_rotary_pos_emb
import torch.nn.functional as F  # Import F




class TestGQAAttention(unittest.TestCase):

    def setUp(self):
        self.config = MistralConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA
            max_position_embeddings=1024,
            sliding_window=None  # Disable Sliding window for most tests.
        )
        self.attention = MistralAttention(self.config)
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        self.attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # torch.device object
        # Use self.device for MistralRotaryEmbedding
        self.attention.rotary_emb = MistralRotaryEmbedding(self.attention.head_dim, max_position_embeddings=self.config.max_position_embeddings, device=self.device)
        self.hidden_states = self.hidden_states.to(self.device)
        self.attention_mask = self.attention_mask.to(self.device)
        self.attention.to(self.device)



    def test_gqa_shapes(self):
        # Test shapes in the forward pass with GQA
        outputs, _, _ = self.attention(self.hidden_states, attention_mask=self.attention_mask)
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Check internal shapes (before repeat_interleave)
        query_states = self.attention.q_proj(self.hidden_states).view(self.batch_size, self.seq_len, self.attention.num_heads, self.attention.head_dim).transpose(1, 2)
        key_states = self.attention.k_proj(self.hidden_states).view(self.batch_size, self.seq_len, self.attention.num_key_value_heads, self.attention.head_dim).transpose(1, 2)
        value_states = self.attention.v_proj(self.hidden_states).view(self.batch_size, self.seq_len, self.attention.num_key_value_heads, self.attention.head_dim).transpose(1, 2)

        self.assertEqual(query_states.shape, (self.batch_size, self.attention.num_heads, self.seq_len, self.attention.head_dim))
        self.assertEqual(key_states.shape, (self.batch_size, self.attention.num_key_value_heads, self.seq_len, self.attention.head_dim))
        self.assertEqual(value_states.shape, (self.batch_size, self.attention.num_key_value_heads, self.seq_len, self.attention.head_dim))
        self.assertEqual(query_states.device, self.device)
        self.assertEqual(key_states.device, self.device)
        self.assertEqual(value_states.device, self.device)



    def test_attention_mask(self):
        # Create a simple attention mask (mask out the last two tokens)
        attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long, device=self.device)
        attention_mask[:, -2:] = 0  # Mask the last two tokens

        # Run attention with the mask
        outputs, _, _ = self.attention(self.hidden_states, attention_mask=attention_mask)

        # Basic shape check
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.config.hidden_size))


    def test_scaled_dot_product_attention_call(self):
        # Test with minimal setup, focusing on the call itself
        batch_size = 2
        num_heads = 8
        seq_len = 10
        head_dim = 64

        # Create dummy query, key, value tensors
        query_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        key_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        value_states = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        attention_mask = torch.ones(batch_size, seq_len, device=self.device).bool()  # Mock attention mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # Correct dimension

        # Call scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False
        )

        # Check output shape
        self.assertEqual(attn_output.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertTrue(attn_output.dtype == query_states.dtype)
        self.assertEqual(attn_output.device, self.device)

    def test_no_sliding_window(self):
        outputs, _, _ = self.attention(self.hidden_states)
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_sliding_window(self):
        # Enable sliding window
        self.config.sliding_window = 4
        self.attention = MistralAttention(self.config)
        self.attention.rotary_emb = MistralRotaryEmbedding(self.attention.head_dim, max_position_embeddings=self.config.max_position_embeddings, device=self.device) # Use self.device
        self.attention.to(self.device)  # Move to the correct device

        outputs, _, _ = self.attention(self.hidden_states, attention_mask=self.attention_mask)
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.config.hidden_size))


    def test_different_q_k_v_lengths(self):
        # Test with different query and key/value sequence lengths
        q_len = 5
        kv_len = 10

        hidden_states_q = torch.randn(self.batch_size, q_len, self.config.hidden_size, device=self.device)
        hidden_states_kv = torch.randn(self.batch_size, kv_len, self.config.hidden_size, device=self.device)

        # Create query, key and value projections from different hidden states
        query_states = self.attention.q_proj(hidden_states_q).view(
            self.batch_size, q_len, self.attention.num_heads, self.attention.head_dim
        ).transpose(1, 2)
        key_states = self.attention.k_proj(hidden_states_kv).view(
            self.batch_size, kv_len, self.attention.num_key_value_heads, self.attention.head_dim
        ).transpose(1, 2)
        value_states = self.attention.v_proj(hidden_states_kv).view(
            self.batch_size, kv_len, self.attention.num_key_value_heads, self.attention.head_dim
        ).transpose(1, 2)

        # Get rotary embeddings
        cos, sin = self.attention.rotary_emb(value_states, seq_len=kv_len)
        # Create separate position_ids for query and key/value
        position_ids_q = torch.arange(0, q_len, dtype=torch.long, device=query_states.device).unsqueeze(0)
        position_ids_k = torch.arange(0, kv_len, dtype=torch.long, device=query_states.device).unsqueeze(0)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids_q, position_ids_k
        )

        # Since key/value are projected with num_key_value_heads (e.g. 8) and query with num_heads (e.g. 8 or 32),
        # we need to repeat key and value along the head dimension to match query.
        num_key_value_groups = self.attention.num_heads // self.attention.num_key_value_heads
        key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

        # Create attention mask taking into consideration the q_len and kv_len.
        # NOTE: F.scaled_dot_product_attention expects attn_mask that broadcast to
        # [batch, num_heads, q_len, kv_len]. Here, we explicitly create a mask with shape [batch, 1, q_len, kv_len]
        # so it broadcasts correctly.
        attention_mask = torch.zeros(self.batch_size, 1, q_len, kv_len, device=self.device).bool()

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False
        )
        self.assertEqual(
            attn_output.shape,
            (self.batch_size, self.attention.num_heads, q_len, self.attention.head_dim)
        )

    def test_full_forward_pass_with_mask(self):
        # Test the full forward pass with a more complex mask
        attention_mask = torch.randint(0, 2, (self.batch_size, self.seq_len), dtype=torch.long, device=self.device) # Random mask
        outputs, _, _ = self.attention(self.hidden_states, attention_mask = attention_mask)
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_edge_cases(self):
      # Batch size of 1
      hidden_states_bs1 = torch.randn(1, self.seq_len, self.config.hidden_size, device=self.device)
      attention_mask_bs1 = torch.ones(1, self.seq_len, dtype=torch.long, device=self.device)
      outputs, _, _ = self.attention(hidden_states_bs1, attention_mask=attention_mask_bs1)
      self.assertEqual(outputs.shape, (1, self.seq_len, self.config.hidden_size))

      # Sequence length of 1
      hidden_states_seq1 = torch.randn(self.batch_size, 1, self.config.hidden_size, device=self.device)
      attention_mask_seq1 = torch.ones(self.batch_size, 1, dtype=torch.long, device=self.device)
      outputs, _, _ = self.attention(hidden_states_seq1, attention_mask=attention_mask_seq1)
      self.assertEqual(outputs.shape, (self.batch_size, 1, self.config.hidden_size))


if __name__ == '__main__':
    unittest.main()