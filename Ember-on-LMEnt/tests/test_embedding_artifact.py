import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNMF_ROOT = PROJECT_ROOT / "external" / "snmf"
if str(SNMF_ROOT) not in sys.path:
    sys.path.insert(0, str(SNMF_ROOT))

from ember.erasure.embed_edit import apply_concept_embed_edit_factored
from ember.erasure.features import load_embedding_artifact, load_embedding_payload


class _LegacyTokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [{1: "foo", 2: "bar"}[int(token_id)] for token_id in token_ids]


class EmbeddingArtifactTests(unittest.TestCase):
    def test_loads_id_native_artifact(self) -> None:
        nmf = SimpleNamespace(
            F_=torch.arange(6, dtype=torch.float32).reshape(3, 2),
            G_=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        )
        payload = {
            "artifact_version": 2,
            "model_key": "tiny-olmo2",
            "nmf": nmf,
            "vprime_token_ids": [1, 3, 4, 6],
            "token_roles": {1: 1, 3: 2, 4: 0, 6: 1},
            "tokenizer_size": 7,
            "embedding_vocab_size": 9,
            "embedding_dim": 3,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embedding.pkl"
            with path.open("wb") as handle:
                pickle.dump(payload, handle)

            artifact = load_embedding_artifact(path)

        self.assertEqual(artifact.version, 2)
        self.assertEqual(artifact.model_key, "tiny-olmo2")
        self.assertEqual(artifact.vprime_token_ids, [1, 3, 4, 6])
        self.assertEqual(artifact.token_roles, {1: 1, 3: 2, 4: 0, 6: 1})
        self.assertEqual(artifact.concept_token_ids, {1, 6})
        self.assertEqual(tuple(artifact.F_dir.shape), (3, 2))
        self.assertEqual(tuple(artifact.G_tok.shape), (4, 2))

    def test_released_legacy_artifact_still_loads(self) -> None:
        nmf = SimpleNamespace(
            F_=torch.ones((3, 2), dtype=torch.float32),
            G_=torch.ones((2, 2), dtype=torch.float32),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embedding.pkl"
            with path.open("wb") as handle:
                pickle.dump({"nmf": nmf, "vprime_token_ids": [2, 5]}, handle)

            artifact = load_embedding_artifact(path)
            F_dir, G_tok, token_ids = load_embedding_payload(path)

        self.assertEqual(artifact.version, 1)
        self.assertIsNone(artifact.token_roles)
        self.assertEqual(artifact.concept_token_ids, set())
        self.assertTrue(torch.equal(F_dir, artifact.F_dir))
        self.assertTrue(torch.equal(G_tok, artifact.G_tok))
        self.assertEqual(token_ids, [2, 5])

    def test_legacy_csv_eligibility_and_gemma_scaling_still_apply(self) -> None:
        model = Olmo2ForCausalLM(Olmo2Config(
            vocab_size=4,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        ))
        nmf = SimpleNamespace(
            F_=torch.ones((8, 1), dtype=torch.float32),
            G_=torch.tensor([[2.0], [3.0]], dtype=torch.float32),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "embedding.pkl"
            potential_path = root / "potential_features.csv"
            with artifact_path.open("wb") as handle:
                pickle.dump({
                    "nmf": nmf,
                    "vprime_token_ids": [1, 2],
                }, handle)
            pd.DataFrame([{"feature": 0, "metric_score": 3.0}]).to_csv(
                potential_path, index=False)
            before = model.get_input_embeddings().weight.detach().clone()

            with (
                patch(
                    "ember.erasure.features._embedding_paths",
                    return_value=(artifact_path, potential_path),
                ),
                patch(
                    "ember.erasure.features.load_token_label_map",
                    return_value={"foo": "Concept", "bar": "Neutral"},
                ),
            ):
                result = apply_concept_embed_edit_factored(
                    model,
                    "gemma-test",
                    "Concept",
                    delta_embed=1.0,
                    rank=1,
                    seed=42,
                    tokenizer=_LegacyTokenizer(),
                )

        after = model.get_input_embeddings().weight.detach()
        self.assertEqual(result["edited_token_ids"], [1])
        expected = before[1] - torch.full((8,), 2.0 / math.sqrt(8))
        self.assertTrue(torch.allclose(after[1], expected))
        self.assertTrue(torch.equal(after[2], before[2]))


if __name__ == "__main__":
    unittest.main()
