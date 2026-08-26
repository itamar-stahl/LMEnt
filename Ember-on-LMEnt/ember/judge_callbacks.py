"""Provider-neutral callback seams for LMEnt feature and Alpaca judging."""
from __future__ import annotations


def describe_feature(prompt: str) -> str:
    """Return a plain-text feature description for the complete prompt."""
    raise NotImplementedError(
        "No feature-description judge is configured. Implement "
        "ember.judge_callbacks.describe_feature(prompt) -> str, or pass a "
        "describe_callback to run_lment_pipeline().")


def classify_feature(prompt: str) -> str:
    """Return JSON text: {\"is_member\": bool, \"confidence\": 0..1}."""
    raise NotImplementedError(
        "No feature-classification judge is configured. Implement "
        "ember.judge_callbacks.classify_feature(prompt) -> JSON str, or pass a "
        "classify_callback to run_lment_pipeline().")


def score_alpaca_relevance(prompt: str) -> str:
    """Return text ending in ``Rating: [[0|1|2]]`` for the complete prompt."""
    raise NotImplementedError(
        "No Alpaca relevance judge is configured. Implement "
        "ember.judge_callbacks.score_alpaca_relevance(prompt) -> str, or pass "
        "a relevance_callback to evaluate_lment_alpaca().")


def score_alpaca_fluency(prompt: str) -> str:
    """Return text ending in ``Rating: [[0|1|2]]`` for the complete prompt."""
    raise NotImplementedError(
        "No Alpaca fluency judge is configured. Implement "
        "ember.judge_callbacks.score_alpaca_fluency(prompt) -> str, or pass a "
        "fluency_callback to evaluate_lment_alpaca().")


__all__ = [
    "describe_feature", "classify_feature",
    "score_alpaca_relevance", "score_alpaca_fluency",
]
