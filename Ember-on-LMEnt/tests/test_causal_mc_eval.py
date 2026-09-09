import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ember.evals.causal_mc import evaluate_causal_mc
from ember.evals.mc import prepare_mc_items
from ember.evals.schema import PreparedMCItem
from ember.local_datasets import load_mc_qa_items


class _ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    model_max_length = 32

    _ids = {
        "Question: q\nAnswer:": [1, 2],
        " a": [3],
        " bc": [4, 5],
        " c": [6],
        " d": [7],
    }

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(self._ids[text])}


class _ToyBigramLM(torch.nn.Module):
    def __init__(self, context_logit: float):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 2)
        self.context_logit = context_logit
        self.config = SimpleNamespace(max_position_embeddings=32)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None):
        batch, width = input_ids.shape
        logits = torch.full((batch, width, 8), -5.0, device=input_ids.device)
        for row in range(batch):
            for pos in range(width):
                token = int(input_ids[row, pos])
                if token == 1:
                    logits[row, pos, 2] = self.context_logit
                elif token == 2:
                    logits[row, pos, 3] = 2.0
                    logits[row, pos, 4] = 1.0
                    logits[row, pos, 6] = -2.0
                    logits[row, pos, 7] = -3.0
                elif token == 4:
                    logits[row, pos, 5] = 3.0
        return SimpleNamespace(logits=logits)


class CausalMCEvaluationTests(unittest.TestCase):
    def test_custom_eval_json_filters_concept_and_shuffles_deterministically(self) -> None:
        item = {
            "q": "Which one?",
            "correct_answer": "one",
            "options": ["one", "two", "three", "four"],
        }
        data = {
            "Arbitrary concept": {"QA_train": [item]},
            "Another concept": {"QA_train": [{**item, "q": "Other?"}]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            raw = load_mc_qa_items(
                "qa_train", concept="Arbitrary concept", json_path=path)

        first = prepare_mc_items(raw, seed=17)
        second = prepare_mc_items(raw, seed=17)
        self.assertEqual(len(first), 1)
        self.assertEqual(first, second)
        self.assertEqual(first[0].concept, "Arbitrary concept")

    def test_scores_only_continuations_with_per_character_accuracy_and_margin(self) -> None:
        item = PreparedMCItem(
            concept="Arbitrary concept",
            subset="QA",
            split="train",
            question="q",
            options_by_letter={"A": "a", "B": "bc", "C": "c", "D": "d"},
            correct_letter="A",
            correct_text="a",
        )

        low_context = evaluate_causal_mc(
            _ToyBigramLM(context_logit=-100.0), _ToyTokenizer(), [item])
        high_context = evaluate_causal_mc(
            _ToyBigramLM(context_logit=100.0), _ToyTokenizer(), [item])

        self.assertEqual(low_context.accuracy, 1.0)
        self.assertAlmostEqual(low_context.mean_margin, 0.278444, places=5)
        self.assertEqual(low_context.records[0]["prediction"], "A")
        self.assertEqual(low_context.records[0]["scores_per_char"],
                         high_context.records[0]["scores_per_char"])


if __name__ == "__main__":
    unittest.main()
