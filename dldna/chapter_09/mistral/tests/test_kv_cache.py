import unittest
import torch
from dldna.chapter_09.mistral.simple_mistral import MistralAttention, MistralConfig, MistralRotaryEmbedding, apply_rotary_pos_emb, repeat_kv

class TestKVCache(unittest.TestCase):

    def setUp(self):
        self.config = MistralConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA
            max_position_embeddings=1024,
            use_cache=True  # Ensure caching is enabled
        )
        self.attention = MistralAttention(self.config)
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        self.attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)
        self.attention.rotary_emb = MistralRotaryEmbedding(self.attention.head_dim, max_position_embeddings=self.config.max_position_embeddings) # For consistency

    def test_no_cache(self):
        # Run without cache
        self.config.use_cache = False
        self.attention = MistralAttention(self.config) # Re-init to apply use_cache change.
        self.attention.rotary_emb = MistralRotaryEmbedding(self.attention.head_dim, max_position_embeddings=self.config.max_position_embeddings) # For consistency

        outputs, _, past_key_value = self.attention(self.hidden_states, attention_mask=self.attention_mask)
        self.assertIsNone(past_key_value)

    def test_cache_initial(self):
        # Run with cache (initial step)
        outputs, _, past_key_value = self.attention(self.hidden_states, attention_mask=self.attention_mask)
        self.assertIsNotNone(past_key_value)
        self.assertEqual(len(past_key_value), 2)  # (key, value)
        self.assertEqual(past_key_value[0].shape, (self.batch_size, self.config.num_key_value_heads, self.seq_len, self.attention.head_dim))
        self.assertEqual(past_key_value[1].shape, (self.batch_size, self.config.num_key_value_heads, self.seq_len, self.attention.head_dim))


    def test_cache_subsequent(self):
      # Run with cache (initial step)
      _, _, past_key_value = self.attention(self.hidden_states, attention_mask=self.attention_mask)

      # Run with cache (subsequent step)
      next_seq_len = 5
      next_hidden_states = torch.randn(self.batch_size, next_seq_len, self.config.hidden_size)
      next_attention_mask = torch.ones(self.batch_size, next_seq_len, dtype=torch.long)

      outputs, _, new_past_key_value = self.attention(next_hidden_states, attention_mask=next_attention_mask, past_key_value=past_key_value)
      self.assertIsNotNone(new_past_key_value)
      self.assertEqual(len(new_past_key_value), 2)
      self.assertEqual(new_past_key_value[0].shape, (self.batch_size, self.config.num_key_value_heads, self.seq_len + next_seq_len, self.attention.head_dim))
      self.assertEqual(new_past_key_value[1].shape, (self.batch_size, self.config.num_key_value_heads, self.seq_len + next_seq_len, self.attention.head_dim))
      # Verify that the cache has been updated correctly.
      self.assertTrue(torch.allclose(new_past_key_value[0][:, :, :self.seq_len, :], past_key_value[0]))
      self.assertTrue(torch.allclose(new_past_key_value[1][:, :, :self.seq_len, :], past_key_value[1]))

    def test_repeat_kv(self):
        # Test the repeat_kv function
        num_kv_heads = self.config.num_key_value_heads
        n_rep = self.config.num_attention_heads // num_kv_heads
        kv_states = torch.randn(self.batch_size, num_kv_heads, self.seq_len, self.attention.head_dim)

        repeated_kv_states = repeat_kv(kv_states, n_rep)
        self.assertEqual(repeated_kv_states.shape, (self.batch_size, self.config.num_attention_heads, self.seq_len, self.attention.head_dim))

        # Verify content of repeated tensor
        for i in range(n_rep):
            self.assertTrue(torch.allclose(repeated_kv_states[:, i::n_rep, :, :], kv_states))

if __name__ == '__main__':
    unittest.main()