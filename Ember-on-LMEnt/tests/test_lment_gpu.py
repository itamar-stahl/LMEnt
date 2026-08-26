import gc
import os
import tempfile
import unittest
from pathlib import Path

import torch

from ember.erasure.model_loader import load_local_causal_lm
from ember.evals.causal_mc import evaluate_causal_mc
from ember.evals.mc import prepare_mc_items
from ember.local_datasets import load_mc_qa_items
from tests.test_hf_embedding_features import _write_tiny_olmo2_checkpoint


CUDA_AVAILABLE = torch.cuda.is_available()
REAL_MODEL_PATH = os.environ.get("EMBER_LMENT_MODEL_PATH")


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is not available")
class LMEntCudaTests(unittest.TestCase):
    def tearDown(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

    def test_tiny_olmo2_loads_and_runs_on_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            _write_tiny_olmo2_checkpoint(checkpoint)
            model, tokenizer = load_local_causal_lm(
                checkpoint, device="auto", dtype=torch.float32)

            input_ids = torch.tensor(
                [[tokenizer.bos_token_id, 5]], dtype=torch.long, device="cuda")
            logits = model(input_ids=input_ids, use_cache=False).logits

        self.assertEqual(model.get_input_embeddings().weight.device.type, "cuda")
        self.assertEqual(logits.device.type, "cuda")
        self.assertTrue(torch.isfinite(logits).all())

    @unittest.skipUnless(
        REAL_MODEL_PATH,
        "Set EMBER_LMENT_MODEL_PATH to run the full-checkpoint CUDA test",
    )
    def test_real_lment_checkpoint_scores_on_cuda(self) -> None:
        checkpoint = Path(REAL_MODEL_PATH).resolve()
        self.assertTrue((checkpoint / "config.json").is_file(), checkpoint)
        model, tokenizer = load_local_causal_lm(
            checkpoint, device="auto", dtype=torch.float32)
        try:
            embedding = model.get_input_embeddings().weight
            output = model.get_output_embeddings().weight
            self.assertEqual(embedding.device.type, "cuda")
            self.assertEqual(tuple(embedding.shape), (100352, 2048))
            self.assertEqual(embedding.dtype, torch.float32)
            self.assertNotEqual(embedding.data_ptr(), output.data_ptr())

            item = prepare_mc_items(load_mc_qa_items(
                "qa_train", concept="Culture of Greece")[:1])
            evaluation = evaluate_causal_mc(model, tokenizer, item)
            self.assertEqual(evaluation.n, 1)
            self.assertTrue(torch.isfinite(torch.tensor(
                evaluation.records[0]["scores_per_char"])).all())
        finally:
            del model
            del tokenizer


if __name__ == "__main__":
    unittest.main()
