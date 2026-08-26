"""Alpaca prompt-continuation evaluation for base LMEnt models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from ember.evals.alpaca import evaluate_alpaca
from ember.evals.callback_judge import CallbackAlpacaEvaluator, TextCallback
from ember.evals.model_wrap import WrappedHFModel


@dataclass(frozen=True)
class GPUProfile:
    name: str
    expected_name_fragment: str
    max_batch_size: int
    dtype: str


GPU_PROFILES = {
    "rtx5070-laptop": GPUProfile(
        name="rtx5070-laptop",
        expected_name_fragment="RTX 5070 Laptop",
        max_batch_size=1,
        dtype="fp32",
    ),
    "h100": GPUProfile(
        name="h100",
        expected_name_fragment="H100",
        max_batch_size=32,
        dtype="bf16",
    ),
}

_PROFILE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def require_gpu_profile(name: str) -> GPUProfile:
    if name not in GPU_PROFILES:
        raise ValueError(f"gpu_type must be one of: {sorted(GPU_PROFILES)}")
    if not torch.cuda.is_available():
        raise RuntimeError("Alpaca evaluation requires CUDA")
    profile = GPU_PROFILES[name]
    actual = torch.cuda.get_device_name(0)
    if profile.expected_name_fragment.lower() not in actual.lower():
        raise RuntimeError(
            f"GPU profile {name!r} expects a device containing "
            f"{profile.expected_name_fragment!r}, but CUDA reports {actual!r}")
    return profile


def _available_batch_size(profile: GPUProfile) -> tuple[int, float, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    gib = 1024 ** 3
    # Reserve about 2 GiB of currently free VRAM per generated item. The
    # hardware profile remains a hard upper bound.
    memory_bound = max(1, int(free_bytes // (2 * gib)))
    return (
        min(profile.max_batch_size, memory_bound),
        round(free_bytes / gib, 3),
        round(total_bytes / gib, 3),
    )


def evaluate_lment_prompts(*, model: Any, tokenizer: Any,
                           prompts: Sequence[str], gpu_type: str,
                           relevance_callback: TextCallback,
                           fluency_callback: TextCallback,
                           max_new_tokens: int = 200) -> Dict[str, Any]:
    """Generate raw continuations and score them through complete-prompt callbacks."""
    profile = require_gpu_profile(gpu_type)
    actual_dtype = model.get_input_embeddings().weight.dtype
    if actual_dtype != _PROFILE_DTYPES[profile.dtype]:
        raise ValueError(
            f"GPU profile {gpu_type!r} requires model dtype {profile.dtype}, "
            f"but the model uses {actual_dtype}")
    batch_size, free_vram_gib, total_vram_gib = _available_batch_size(profile)
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
        "gpu_profile": asdict(profile),
        "resolved_batch_size": batch_size,
        "free_vram_gib_at_start": free_vram_gib,
        "total_vram_gib": total_vram_gib,
        "n": n,
        "mean_relevance": sum(relevance_scores) / n if n else 0.0,
        "mean_fluency": sum(fluency_scores) / n if n else 0.0,
        "records": records,
    }


def evaluate_lment_alpaca(*, model: Any, tokenizer: Any, split: str,
                          gpu_type: str, relevance_callback: TextCallback,
                          fluency_callback: TextCallback,
                          max_items: Optional[int] = None) -> Dict[str, Any]:
    """Run the repository Alpaca split as raw LMEnt prompt continuations."""
    profile = require_gpu_profile(gpu_type)
    actual_dtype = model.get_input_embeddings().weight.dtype
    if actual_dtype != _PROFILE_DTYPES[profile.dtype]:
        raise ValueError(
            f"GPU profile {gpu_type!r} requires model dtype {profile.dtype}, "
            f"but the model uses {actual_dtype}")
    batch_size, free_vram_gib, total_vram_gib = _available_batch_size(profile)
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
        "gpu_profile": asdict(profile),
        "resolved_batch_size": batch_size,
        "free_vram_gib_at_start": free_vram_gib,
        "total_vram_gib": total_vram_gib,
        "n": n,
        "mean_relevance": sum(relevance) / n if n else 0.0,
        "mean_fluency": sum(fluency) / n if n else 0.0,
        "records": records,
    }


__all__ = [
    "GPUProfile", "GPU_PROFILES", "evaluate_lment_alpaca",
    "evaluate_lment_prompts", "require_gpu_profile",
]
