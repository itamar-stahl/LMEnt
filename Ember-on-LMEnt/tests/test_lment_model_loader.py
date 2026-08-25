import tempfile
import unittest
from pathlib import Path

import torch

from ember.erasure.model_loader import load_local_causal_lm
from tests.test_hf_embedding_features import _write_tiny_olmo2_checkpoint


class LMEntModelLoaderTests(unittest.TestCase):
    def test_local_loader_preserves_checkpoint_dtype_tokens_cache_and_untied_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "tiny-olmo2"
            model_path.mkdir()
            _write_tiny_olmo2_checkpoint(model_path)

            model, tokenizer = load_local_causal_lm(
                model_path,
                dtype=torch.float32,
                device="cpu",
            )

        self.assertEqual(model.get_input_embeddings().weight.dtype, torch.float32)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.bos_token_id, 1)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertTrue(model.config.use_cache)
        self.assertEqual(model.config.max_position_embeddings, 32)
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertNotEqual(
            model.get_input_embeddings().weight.data_ptr(),
            model.get_output_embeddings().weight.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
