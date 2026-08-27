import unittest
from unittest.mock import patch

import torch

from ember.evals.callback_judge import CallbackAlpacaEvaluator
from ember.evals.alpaca import evaluate_alpaca
from ember.evals.lment_alpaca import require_cuda


class LMEntAlpacaTests(unittest.TestCase):
    def test_alpaca_smoke_limit_bounds_generation_and_judging(self) -> None:
        class Model:
            def generate_multiple(self, prompts, **_kwargs):
                self.prompts = prompts
                return ["completion"] * len(prompts)

        class Evaluator:
            def score_alpaca_instruct(self, _instruction, _completion):
                return 2

            def score_alpaca_fluency(self, _completion):
                return 1

        model = Model()
        with (
            patch("ember.evals.alpaca.load_alpaca_indices", return_value=[0, 1, 2]),
            patch("ember.evals.alpaca.load_alpaca_eval_local", return_value=[
                {"instruction": "first"},
                {"instruction": "second"},
                {"instruction": "third"},
            ]),
        ):
            relevance, fluency, records = evaluate_alpaca(
                model, Evaluator(), "test", strict=True, max_items=1)

        self.assertEqual(model.prompts, ["first"])
        self.assertEqual(relevance, [2])
        self.assertEqual(fluency, [1])
        self.assertEqual(len(records), 1)

    def test_callbacks_receive_complete_prompts_and_return_valid_scores(self) -> None:
        prompts = []

        def callback(prompt: str) -> str:
            prompts.append(prompt)
            return "Reason\nRating: [[2]]"

        evaluator = CallbackAlpacaEvaluator(callback, callback)
        self.assertEqual(evaluator.score_alpaca_instruct("Continue this", "text"), 2)
        self.assertEqual(evaluator.score_alpaca_fluency("text"), 2)
        self.assertIn("[Instruction Start]\nContinue this", prompts[0])
        self.assertIn("[Completion Start]\ntext", prompts[0])
        self.assertIn("[Sentence Fragment Start]", prompts[1])

    def test_callback_rating_outside_scale_is_rejected(self) -> None:
        evaluator = CallbackAlpacaEvaluator(
            lambda _prompt: "Rating: [[5]]",
            lambda _prompt: "Rating: [[2]]",
        )
        with self.assertRaisesRegex(ValueError, "0, 1, or 2"):
            evaluator.score_alpaca_instruct("prompt", "completion")

    def test_cuda_is_required_without_a_hardware_profile(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=True):
            require_cuda()
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                require_cuda()


if __name__ == "__main__":
    unittest.main()
