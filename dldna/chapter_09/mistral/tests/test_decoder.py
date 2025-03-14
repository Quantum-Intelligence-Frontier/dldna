import unittest
import torch
import torch.nn as nn
from dldna.chapter_09.mistral.simple_mistral  import MistralConfig, MistralForCausalLM

class TestMistralDecoder(unittest.TestCase):
    def setUp(self):
        # Use a small, simple configuration for testing (reduce computation)
        self.config = MistralConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,  # Verify 4 % 1 == 0
            max_position_embeddings=128,
            sliding_window=8,
        )
        self.model = MistralForCausalLM(self.config)
        # Set seed for reproducibility
        torch.manual_seed(42)

    def test_forward_no_labels(self):
        """
        When calling forward without labels, check if the shape of logits is
        (batch_size, seq_length, vocab_size).
        """
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.float32)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        self.assertIn("logits", outputs)
        logits = outputs["logits"]
        self.assertEqual(
            logits.shape,
            (batch_size, seq_length, self.config.vocab_size),
            "The shape of logits is different from expected."
        )

    def test_prepare_inputs_for_generation_with_past(self):
        """Test if prepare_inputs_for_generation works correctly when past cache (past_key_values) is provided."""
        batch_size = 1
        seq_length = 5
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        # When creating dummy_past, the second dimension should use num_key_value_heads.
        dummy_past = []
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        for _ in range(self.config.num_hidden_layers):
            key_tensor = torch.randn(batch_size, self.config.num_key_value_heads, 1, head_dim)
            value_tensor = torch.randn(batch_size, self.config.num_key_value_heads, 1, head_dim)
            dummy_past.append((key_tensor, value_tensor))

        inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=dummy_past,
            attention_mask=torch.ones(batch_size, seq_length)
        )
        # When cache is present, input_ids should contain only the last token.
        self.assertIn("input_ids", inputs)
        self.assertEqual(
            inputs["input_ids"].shape[-1], 1,
            "When past_key_values is present, the length of input_ids should be 1."
        )


    def test_forward_with_labels_and_backward(self):
        """
        Test if the loss is calculated and backpropagation works correctly when labels are provided.
        """
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        # Generally, during training, input_ids are shifted to be used as targets.
        labels = input_ids.clone()
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.float32)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        self.assertIn("loss", outputs)
        loss = outputs["loss"]
        # Check if gradients are calculated without errors during backpropagation
        loss.backward()

        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(
            any(grad is not None for grad in grads),
            "Gradients do not exist for the model's parameters."
        )

    def test_prepare_inputs_for_generation_without_inputs_embeds(self):
        """
        Test if the input remains unchanged when only input_ids are passed without inputs_embeds.
        """
        batch_size = 1
        seq_length = 3
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.float32)

        inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask
        )

        self.assertIn("input_ids", inputs)
        # When past_key_values is not present, input_ids should remain unchanged
        self.assertTrue(
            torch.equal(inputs["input_ids"], input_ids),
            "When past_key_values is not present, input_ids should not be changed."
        )

if __name__ == "__main__":
    unittest.main()