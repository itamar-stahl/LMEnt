import json
import unittest

import torch

from ember.gemma_judge import GemmaJudge


class _FakeBatch(dict):
    def to(self, device):
        self["input_ids"] = self["input_ids"].to(device)
        return self


class _FakeTokenizer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        return _FakeBatch(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

    def decode(self, _tokens, **_kwargs):
        return self.responses.pop(0)


class _FakeModel:
    device = torch.device("cpu")

    def generate(self, input_ids, **_kwargs):
        suffix = torch.tensor([[3]], dtype=torch.long, device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


class GemmaJudgeTests(unittest.TestCase):
    def test_public_callbacks_normalize_hosted_model_responses(self) -> None:
        tokenizer = _FakeTokenizer([
            "A concise feature description.",
            "Answer follows: {\"is_member\": true, \"confidence\": 0.91}",
            "The completion is relevant. Rating: [[2]]",
            "The completion is awkward. Rating: [[1]]",
        ])
        judge = GemmaJudge(
            model=_FakeModel(), tokenizer=tokenizer, device="cpu",
            max_new_tokens=64,
        )

        self.assertEqual(
            judge.describe_feature("FULL DESCRIPTION PROMPT"),
            "A concise feature description.",
        )
        self.assertEqual(
            json.loads(judge.classify_feature("FULL CLASSIFICATION PROMPT")),
            {"is_member": True, "confidence": 0.91},
        )
        self.assertEqual(
            judge.score_alpaca_relevance("FULL RELEVANCE PROMPT"),
            "Rating: [[2]]",
        )
        self.assertEqual(
            judge.score_alpaca_fluency("FULL FLUENCY PROMPT"),
            "Rating: [[1]]",
        )
        self.assertEqual(tokenizer.prompts, [
            "FULL DESCRIPTION PROMPT",
            "FULL CLASSIFICATION PROMPT",
            "FULL RELEVANCE PROMPT",
            "FULL FLUENCY PROMPT",
        ])

    def test_invalid_structured_response_stops_the_run(self) -> None:
        judge = GemmaJudge(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(["not JSON"]),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "JSON object"):
            judge.classify_feature("prompt")


if __name__ == "__main__":
    unittest.main()
