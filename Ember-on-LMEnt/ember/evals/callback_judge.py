"""Provider-neutral string callbacks for Alpaca relevance and fluency prompts."""
from __future__ import annotations

from typing import Callable

from ember.evals.gemini import GeminiEvaluator


TextCallback = Callable[[str], str]


class CallbackAlpacaEvaluator:
    def __init__(self, relevance_callback: TextCallback,
                 fluency_callback: TextCallback) -> None:
        self.relevance_callback = relevance_callback
        self.fluency_callback = fluency_callback

    @staticmethod
    def _rating(response: str) -> int:
        if not isinstance(response, str):
            raise ValueError("Alpaca judge callback must return a string")
        rating = GeminiEvaluator._parse_rating(response)
        if rating not in {0, 1, 2}:
            raise ValueError(f"Alpaca judge rating must be 0, 1, or 2; got {rating}")
        return rating

    def score_alpaca_instruct(self, instruction: str, completion: str) -> int:
        if not completion.strip():
            return 0
        prompt = GeminiEvaluator.ALPACA_INSTRUCT_PROMPT.format(
            instructions=instruction, completion=completion)
        return self._rating(self.relevance_callback(prompt))

    def score_alpaca_fluency(self, completion: str) -> int:
        if not completion.strip():
            return 0
        prompt = GeminiEvaluator.ALPACA_FLUENCY_PROMPT.format(completion=completion)
        return self._rating(self.fluency_callback(prompt))


__all__ = ["CallbackAlpacaEvaluator", "TextCallback"]
