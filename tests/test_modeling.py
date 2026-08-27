import os
import unittest

import torch

from eshmun.models.eshmun import EshmunConfig, EshmunForCausalLM, EshmunModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_2000 = os.path.join(
    REPO_ROOT, "data", "checkpoints", "kothar-pretrain-409m", "checkpoint-2000"
)


def tiny_config() -> EshmunConfig:
    return EshmunConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        ffn_dim=64,
        num_attention_heads=4,
        max_position_embeddings=32,
        word_embed_proj_dim=32,
        dropout=0.0,
        attention_dropout=0.0,
        layerdrop=0.0,
    )


class EshmunForCausalLMTests(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.model = EshmunForCausalLM(self.config)
        self.model.eval()

    def test_forward_logits_shape(self):
        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        self.assertEqual(
            outputs.logits.shape, (2, 10, self.config.vocab_size)
        )

    def test_forward_with_labels_computes_scalar_loss(self):
        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.shape, ())
        self.assertTrue(torch.isfinite(outputs.loss))

    def test_lm_head_is_tied_to_input_embeddings(self):
        self.assertIs(
            self.model.lm_head.weight, self.model.get_input_embeddings().weight
        )

    def test_attention_mask_blocks_masked_position_from_later_tokens(self):
        torch.manual_seed(0)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 8))
        full_mask = torch.ones_like(input_ids)
        masked = full_mask.clone()
        masked[:, 0] = 0  # mask the first token out of every later position's attention

        with torch.no_grad():
            logits_full = self.model(input_ids=input_ids, attention_mask=full_mask).logits
            logits_masked = self.model(input_ids=input_ids, attention_mask=masked).logits

        # Blocking attention to position 0 must change logits at every later
        # position (they can all attend back to it causally), proving
        # attention_mask is actually applied rather than ignored.
        self.assertFalse(
            torch.allclose(logits_full[:, 1:], logits_masked[:, 1:], atol=1e-5)
        )

    def test_generate_extends_sequence_by_requested_length(self):
        torch.manual_seed(0)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 5))
        generated = self.model.generate(
            input_ids=input_ids, max_new_tokens=4, do_sample=False
        )
        self.assertEqual(generated.shape, (1, 9))
        self.assertTrue(torch.equal(generated[:, :5], input_ids))

    def test_eshmun_model_returns_correct_hidden_size(self):
        base_model = EshmunModel(self.config)
        base_model.eval()
        input_ids = torch.randint(0, self.config.vocab_size, (1, 6))
        with torch.no_grad():
            outputs = base_model(input_ids=input_ids)
        self.assertEqual(
            outputs.last_hidden_state.shape, (1, 6, self.config.hidden_size)
        )


@unittest.skipUnless(
    os.path.isdir(CHECKPOINT_2000), f"checkpoint not found at {CHECKPOINT_2000}"
)
class Kothar2000CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EshmunForCausalLM.from_pretrained(
            CHECKPOINT_2000, torch_dtype=torch.float32
        )
        cls.model.eval()

    def test_loads_with_expected_architecture(self):
        self.assertEqual(self.model.config.model_type, "eshmun")
        self.assertEqual(self.model.config.vocab_size, 50289)
        self.assertEqual(self.model.config.hidden_size, 2048)
        self.assertEqual(self.model.config.num_hidden_layers, 6)

    def test_loads_in_float32(self):
        self.assertEqual(self.model.lm_head.weight.dtype, torch.float32)

    def test_forward_pass_on_real_checkpoint(self):
        input_ids = torch.randint(0, self.model.config.vocab_size, (1, 12))
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        self.assertEqual(outputs.logits.shape, (1, 12, self.model.config.vocab_size))
        self.assertTrue(torch.isfinite(outputs.logits).all())


if __name__ == "__main__":
    unittest.main()
