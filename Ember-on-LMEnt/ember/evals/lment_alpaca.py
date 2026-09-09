"""Alpaca prompt-continuation evaluation for base LMEnt models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from ember.evals.alpaca import evaluate_alpaca
from ember.evals.callback_judge import CallbackAlpacaEvaluator, TextCallback
from ember.evals.model_wrap import WrappedHFModel


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Alpaca evaluation requires CUDA")


def _available_batch_size() -> tuple[int, float, float]:
    require_cuda()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    gib = 1024 ** 3
    # Reserve about 2 GiB of currently free VRAM per generated item and cap
    # the batch to avoid unexpectedly large generation allocations.
    memory_bound = max(1, int(free_bytes // (2 * gib)))
    return (
        min(32, memory_bound),
        round(free_bytes / gib, 3),
        round(total_bytes / gib, 3),
    )


def evaluate_lment_prompts(*, model: Any, tokenizer: Any,
                           prompts: Sequence[str],
                           relevance_callback: TextCallback,
                           fluency_callback: TextCallback,
                           max_new_tokens: int = 200) -> Dict[str, Any]:
    """Generate raw continuations and score them through complete-prompt callbacks."""
    batch_size, free_vram_gib, total_vram_gib = _available_batch_size()
    wrapper = WrappedHFModel(model, tokenizer, model_format="lment")
    evaluator = CallbackAlpacaEvaluator(relevance_callback, fluency_callback)
    completions = wrapper.generate_multiple(
        list(prompts), max_new_tokens=max_new_tokens, do_sample=False,
        batch_size=batch_size, verbose=False)
    records: List[Dict[str, Any]] = []
    relevance_scores: List[int] = []
    fluency_scores: List[int] = []
    for prompt, completion in zip(prompts, completions):
        relevance = evaluator.score_alpaca_instruct(prompt, completion)
        fluency = evaluator.score_alpaca_fluency(completion)
        relevance_scores.append(relevance)
        fluency_scores.append(fluency)
        records.append({
            "prompt": prompt,
            "completion": completion,
            "relevance_score": relevance,
            "fluency_score": fluency,
        })
    n = len(records)
    return {
        "kind": "lment_raw_prompt_continuation",
        "gpu": torch.cuda.get_device_name(0),
        "model_dtype": str(model.get_input_embeddings().weight.dtype),
        "resolved_batch_size": batch_size,
        "free_vram_gib_at_start": free_vram_gib,
        "total_vram_gib": total_vram_gib,
        "n": n,
        "mean_relevance": sum(relevance_scores) / n if n else 0.0,
        "mean_fluency": sum(fluency_scores) / n if n else 0.0,
        "records": records,
    }


def evaluate_lment_alpaca(*, model: Any, tokenizer: Any, split: str,
                          relevance_callback: TextCallback,
                          fluency_callback: TextCallback,
                          max_items: Optional[int] = None) -> Dict[str, Any]:
    """Run the repository Alpaca split as raw LMEnt prompt continuations."""
    batch_size, free_vram_gib, total_vram_gib = _available_batch_size()
    wrapper = WrappedHFModel(model, tokenizer, model_format="lment")
    evaluator = CallbackAlpacaEvaluator(relevance_callback, fluency_callback)
    relevance, fluency, records = evaluate_alpaca(
        wrapper, evaluator, split, batch_size=batch_size, strict=True,
        max_items=max_items)
    n = len(records)
    return {
        "kind": "lment_alpaca_raw_prompt_continuation",
        "warning": (
            "LMEnt is a base language model. These are raw prompt continuations, "
            "not instruction-following outputs."),
        "split": split,
        "max_items": max_items,
        "gpu": torch.cuda.get_device_name(0),
        "model_dtype": str(model.get_input_embeddings().weight.dtype),
        "resolved_batch_size": batch_size,
        "free_vram_gib_at_start": free_vram_gib,
        "total_vram_gib": total_vram_gib,
        "n": n,
        "mean_relevance": sum(relevance) / n if n else 0.0,
        "mean_fluency": sum(fluency) / n if n else 0.0,
        "records": records,
    }


__all__ = [
    "evaluate_lment_alpaca", "evaluate_lment_prompts", "require_cuda",
]
