import unittest
import torch
from dldna.chapter_09.mistral.simple_mistral import MistralRotaryEmbedding, apply_rotary_pos_emb, rotate_half


class TestRoPE(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.num_heads = 8
        self.seq_len = 20
        self.head_dim = 64
        self.max_position_embeddings = 32
        self.base = 10000.0
        self.device = "cpu"  # Use CPU for simplicity, can change to "cuda" if available

        self.rotary_emb = MistralRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.base,
            device=self.device
        )

        self.q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        self.k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device)
        self.position_ids = torch.arange(0, self.seq_len, dtype=torch.long, device=self.device).unsqueeze(0)


    def test_rotary_embedding_creation(self):
        self.assertEqual(self.rotary_emb.dim, self.head_dim)
        self.assertEqual(self.rotary_emb.max_position_embeddings, self.max_position_embeddings)
        self.assertEqual(self.rotary_emb.base, self.base)
        self.assertIsInstance(self.rotary_emb.inv_freq, torch.Tensor)

    def test_forward_pass(self):
        cos, sin = self.rotary_emb(self.q, seq_len=self.seq_len)
        self.assertEqual(cos.shape, (1, 1, self.seq_len, self.head_dim))
        self.assertEqual(sin.shape, (1, 1, self.seq_len, self.head_dim))
        self.assertEqual(cos.dtype, self.q.dtype)
        self.assertEqual(sin.dtype, self.q.dtype)

    def test_cos_sin_cache_update(self):
        longer_seq_len = self.seq_len + 10
        initial_long_seq_len = self.max_position_embeddings + 10

        # First forward pass with initial_long_seq_len
        _ = self.rotary_emb(self.q, seq_len=initial_long_seq_len)
        self.assertEqual(self.rotary_emb.max_seq_len_cached, initial_long_seq_len) # Check after initial update

        # Second forward pass with longer_seq_len (which is smaller)
        cos, sin = self.rotary_emb(self.q, seq_len=longer_seq_len)
        # max_seq_len_cached should *not* have changed
        self.assertEqual(self.rotary_emb.max_seq_len_cached, initial_long_seq_len)
        self.assertEqual(cos.shape, (1, 1, longer_seq_len, self.head_dim))
        self.assertEqual(sin.shape, (1, 1, longer_seq_len, self.head_dim))


    def test_rotate_half(self):
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        rotated_x = rotate_half(x)
        self.assertEqual(rotated_x.shape, x.shape)
        # Check if the rotation is done correctly.
        self.assertTrue(torch.allclose(rotated_x[..., :self.head_dim // 2], -x[..., self.head_dim // 2:]))
        self.assertTrue(torch.allclose(rotated_x[..., self.head_dim // 2:], x[..., :self.head_dim // 2]))

    def test_apply_rotary_pos_emb(self):
        cos, sin = self.rotary_emb(self.q, seq_len=self.seq_len)
        q_embed, k_embed = apply_rotary_pos_emb(self.q, self.k, cos, sin, self.position_ids)
        self.assertEqual(q_embed.shape, self.q.shape)
        self.assertEqual(k_embed.shape, self.k.shape)
        self.assertEqual(q_embed.dtype, self.q.dtype)
        self.assertEqual(k_embed.dtype, self.k.dtype)

    def test_apply_rotary_pos_emb_different_seq_len(self):
        # Test with a shorter sequence length
        short_seq_len = self.seq_len // 2
        short_position_ids = torch.arange(0, short_seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        q_short = torch.randn(self.batch_size, self.num_heads, short_seq_len, self.head_dim, device=self.device)
        k_short = torch.randn(self.batch_size, self.num_heads, short_seq_len, self.head_dim, device=self.device)

        cos, sin = self.rotary_emb(q_short, seq_len=short_seq_len)  # Use short_seq_len
        q_embed, k_embed = apply_rotary_pos_emb(q_short, k_short, cos, sin, short_position_ids)
        self.assertEqual(q_embed.shape, q_short.shape)
        self.assertEqual(k_embed.shape, k_short.shape)

if __name__ == '__main__':
    unittest.main()