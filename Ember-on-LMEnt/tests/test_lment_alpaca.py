import unittest
from unittest.mock import patch

import torch

from ember.evals.callback_judge import CallbackAlpacaEvaluator
from ember.evals.lment_alpaca import require_gpu_profile


class LMEntAlpacaTests(unittest.TestCase):
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

    def test_gpu_profile_checks_requested_hardware(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_name", return_value="NVIDIA H100 80GB HBM3"),
        ):
            profile = require_gpu_profile("h100")
        self.assertEqual(profile.max_batch_size, 32)
        with self.assertRaisesRegex(ValueError, "gpu_type"):
            require_gpu_profile("unknown")


if __name__ == "__main__":
    unittest.main()
