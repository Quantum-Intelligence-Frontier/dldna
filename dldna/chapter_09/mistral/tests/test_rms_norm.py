import unittest
import torch
from dldna.chapter_09.mistral.simple_mistral import MistralRMSNorm


class TestMistralRMSNorm(unittest.TestCase):

    def setUp(self):
        self.hidden_size = 512
        self.eps = 1e-6
        self.rms_norm = MistralRMSNorm(self.hidden_size, eps=self.eps)
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size)

    def test_initialization(self):
        self.assertEqual(self.rms_norm.weight.shape, (self.hidden_size,))
        self.assertEqual(self.rms_norm.variance_epsilon, self.eps)
        self.assertTrue(torch.allclose(self.rms_norm.weight, torch.ones(self.hidden_size)))

    def test_forward_pass(self):
        output = self.rms_norm(self.hidden_states)
        self.assertEqual(output.shape, self.hidden_states.shape)
        self.assertEqual(output.dtype, self.hidden_states.dtype)

    def test_forward_pass_dtype_conversion(self):
        # Test with float16 input
        hidden_states_fp16 = self.hidden_states.to(torch.float16)
        output_fp16 = self.rms_norm(hidden_states_fp16)
        self.assertEqual(output_fp16.shape, hidden_states_fp16.shape)
        self.assertEqual(output_fp16.dtype, hidden_states_fp16.dtype) # Corrected assertion

        # Test with bfloat16 input (if supported)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            hidden_states_bf16 = self.hidden_states.to(torch.bfloat16)
            output_bf16 = self.rms_norm(hidden_states_bf16)
            self.assertEqual(output_bf16.shape, hidden_states_bf16.shape)
            self.assertEqual(output_bf16.dtype, hidden_states_bf16.dtype) # Corrected assertion
        else:
            print("bfloat16 is not supported on this device. Skipping bfloat16 test.")
            self.skipTest("bfloat16 is not supported")


    def test_numerical_stability(self):
        small_hidden_states = self.hidden_states * 1e-6
        output_small = self.rms_norm(small_hidden_states)
        self.assertEqual(output_small.shape, small_hidden_states.shape)
        self.assertFalse(torch.isnan(output_small).any())
        self.assertFalse(torch.isinf(output_small).any())

        large_hidden_states = self.hidden_states * 1e6
        output_large = self.rms_norm(large_hidden_states)
        self.assertEqual(output_large.shape, large_hidden_states.shape)
        self.assertFalse(torch.isnan(output_large).any())
        self.assertFalse(torch.isinf(output_large).any())

    def test_zero_input(self):
        zero_hidden_states = torch.zeros_like(self.hidden_states)
        output_zero = self.rms_norm(zero_hidden_states)
        self.assertEqual(output_zero.shape, zero_hidden_states.shape)
        self.assertFalse(torch.isnan(output_zero).any())
        self.assertTrue(torch.allclose(output_zero, torch.zeros_like(output_zero)))


if __name__ == '__main__':
    unittest.main()