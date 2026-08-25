"""Multiple-choice evaluation for base causal language models.

Options are scored as text continuations of ``Question: ...\nAnswer:``. The
primary score is summed continuation log-likelihood divided by character count;
prompt tokens never contribute to an option's score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ember.evals.schema import LETTER_CHOICES, PreparedMCItem


@dataclass(frozen=True)
class CausalMCEvaluation:
    accuracy: float
    mean_margin: float
    records: List[Dict[str, Any]]

    @property
    def n(self) -> int:
        return len(self.records)

    def to_dict(self, *, include_records: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "n": self.n,
            "accuracy": self.accuracy,
            "mean_margin": self.mean_margin,
        }
        if include_records:
            result["records"] = self.records
        return result


def _model_input_device(model: torch.nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    return embeddings.weight.device


def _sequence_limit(model: torch.nn.Module, tokenizer: Any) -> Optional[int]:
    limits: List[int] = []
    model_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    for value in (model_limit, tokenizer_limit):
        if isinstance(value, int) and 0 < value < 1_000_000_000:
            limits.append(value)
    return min(limits) if limits else None


@torch.no_grad()
def continuation_logprobs(
        model: torch.nn.Module,
        tokenizer: Any,
        context: str,
        continuations: Sequence[str],
        *,
        device: Optional[torch.device | str] = None,
) -> List[Tuple[float, int]]:
    """Return ``(summed log-probability, token count)`` per continuation."""
    if not continuations:
        return []
    context_ids = list(tokenizer(context, add_special_tokens=True)["input_ids"])
    if not context_ids:
        raise ValueError("Context tokenized to an empty sequence")

    sequences: List[List[int]] = []
    continuation_lengths: List[int] = []
    for continuation in continuations:
        continuation_ids = list(
            tokenizer(continuation, add_special_tokens=False)["input_ids"])
        if not continuation_ids:
            raise ValueError(f"Continuation tokenized to an empty sequence: {continuation!r}")
        sequences.append(context_ids + continuation_ids)
        continuation_lengths.append(len(continuation_ids))

    limit = _sequence_limit(model, tokenizer)
    longest = max(len(sequence) for sequence in sequences)
    if limit is not None and longest > limit:
        raise ValueError(
            f"Context plus continuation is {longest} tokens, exceeding model limit {limit}")

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    input_ids = torch.full(
        (len(sequences), longest), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        width = len(sequence)
        input_ids[row, :width] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, :width] = 1

    target_device = torch.device(device) if device is not None else _model_input_device(model)
    outputs = model(
        input_ids=input_ids.to(target_device),
        attention_mask=attention_mask.to(target_device),
    )
    logprobs = torch.log_softmax(outputs.logits.float(), dim=-1)

    scores: List[Tuple[float, int]] = []
    for row, (sequence, n_continuation) in enumerate(
            zip(sequences, continuation_lengths)):
        first_target = len(sequence) - n_continuation
        target_ids = torch.tensor(
            sequence[first_target:], dtype=torch.long, device=logprobs.device)
        positions = torch.arange(
            first_target - 1,
            len(sequence) - 1,
            device=logprobs.device,
        )
        score = logprobs[row, positions, target_ids].sum().item()
        scores.append((float(score), n_continuation))
    return scores


def evaluate_causal_mc(
        model: torch.nn.Module,
        tokenizer: Any,
        items: Sequence[PreparedMCItem],
        *,
        device: Optional[torch.device | str] = None,
) -> CausalMCEvaluation:
    """Evaluate prepared MC items using per-character continuation scores."""
    if not items:
        raise ValueError("Cannot evaluate an empty item list")

    records: List[Dict[str, Any]] = []
    for item in items:
        continuations = [f" {item.options_by_letter[letter]}" for letter in LETTER_CHOICES]
        raw_scores = continuation_logprobs(
            model,
            tokenizer,
            f"Question: {item.question}\nAnswer:",
            continuations,
            device=device,
        )
        summed = [score for score, _ in raw_scores]
        per_token = [score / count for score, count in raw_scores]
        per_char = [
            score / max(len(continuation), 1)
            for (score, _), continuation in zip(raw_scores, continuations)
        ]
        prediction_index = max(range(len(LETTER_CHOICES)), key=per_char.__getitem__)
        prediction = LETTER_CHOICES[prediction_index]
        gold_index = LETTER_CHOICES.index(item.correct_letter)
        margin = per_char[gold_index] - max(
            score for index, score in enumerate(per_char) if index != gold_index)

        records.append({
            "concept": item.concept,
            "subset": item.subset,
            "split": item.split,
            "question": item.question,
            "correct_letter": item.correct_letter,
            "prediction": prediction,
            "is_correct": prediction == item.correct_letter,
            "margin": float(margin),
            "scores_sum": summed,
            "scores_per_token": per_token,
            "scores_per_char": per_char,
        })

    return CausalMCEvaluation(
        accuracy=sum(record["is_correct"] for record in records) / len(records),
        mean_margin=sum(record["margin"] for record in records) / len(records),
        records=records,
    )


__all__ = [
    "CausalMCEvaluation", "continuation_logprobs", "evaluate_causal_mc",
]
