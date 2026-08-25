import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Olmo2Config, Olmo2ForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNMF_ROOT = PROJECT_ROOT / "external" / "snmf"
if str(SNMF_ROOT) not in sys.path:
    sys.path.insert(0, str(SNMF_ROOT))

from ember.erasure.embed_edit import apply_embedding_artifact
from ember.erasure.features import EmbeddingArtifact


class _TokenizerStub:
    all_special_ids = [0]
    added_tokens_decoder = {
        4: SimpleNamespace(special=True),
        5: SimpleNamespace(special=False),
    }

    def __len__(self) -> int:
        return 7

    def convert_ids_to_tokens(self, token_ids):
        return [f"t{token_id}" for token_id in token_ids]


class IdSafeEmbeddingEditTests(unittest.TestCase):
    def test_nonzero_edit_changes_only_concept_ids_with_active_selected_features(self) -> None:
        torch.manual_seed(4)
        model = Olmo2ForCausalLM(Olmo2Config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        ))
        tokenizer = _TokenizerStub()
        direction = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)
        artifact = EmbeddingArtifact(
            F_dir=direction,
            G_tok=torch.tensor([[2.0], [3.0], [4.0], [5.0], [6.0], [0.0]]),
            vprime_token_ids=[1, 2, 3, 0, 4, 6],
            version=2,
            model_key="tiny-olmo2",
            token_roles={1: 1, 2: 2, 3: 0, 0: 1, 4: 1, 6: 1},
            tokenizer_size=7,
            embedding_vocab_size=8,
            embedding_dim=8,
        )
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

        result = apply_embedding_artifact(
            model=model,
            tokenizer=tokenizer,
            artifact=artifact,
            feature_ids=[0],
            delta=0.5,
            model_key="tiny-olmo2",
        )

        after = model.state_dict()
        embedding_name = "model.embed_tokens.weight"
        self.assertEqual(result["edited_token_ids"], [1])
        self.assertEqual(result["n_tokens_edited"], 1)
        self.assertTrue(torch.equal(after[embedding_name][1], before[embedding_name][1] - direction[:, 0]))
        unchanged_rows = torch.tensor([0, 2, 3, 4, 5, 6, 7])
        self.assertTrue(torch.equal(
            after[embedding_name].index_select(0, unchanged_rows),
            before[embedding_name].index_select(0, unchanged_rows),
        ))
        for name, tensor in after.items():
            if name != embedding_name:
                self.assertTrue(torch.equal(tensor, before[name]), msg=name)

    def test_delta_zero_is_an_exact_no_op(self) -> None:
        model = Olmo2ForCausalLM(Olmo2Config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        ))
        artifact = EmbeddingArtifact(
            F_dir=torch.ones((8, 1)),
            G_tok=torch.ones((1, 1)),
            vprime_token_ids=[1],
            version=2,
            model_key="tiny-olmo2",
            token_roles={1: 1},
            tokenizer_size=7,
            embedding_vocab_size=8,
            embedding_dim=8,
        )
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

        result = apply_embedding_artifact(
            model=model,
            tokenizer=_TokenizerStub(),
            artifact=artifact,
            feature_ids=[0],
            delta=0.0,
            model_key="tiny-olmo2",
        )

        self.assertEqual(result["edited_token_ids"], [])
        self.assertEqual(result["n_tokens_edited"], 0)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
