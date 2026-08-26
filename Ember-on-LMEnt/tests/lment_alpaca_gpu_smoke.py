#!/usr/bin/env python3
"""Real-GPU LMEnt raw-continuation smoke with deterministic fake judges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ember.erased_embedding import load_lment_with_erased_embeddings
from ember.erasure.model_loader import pick_dtype
from ember.evals.lment_alpaca import (
    GPU_PROFILES,
    evaluate_lment_prompts,
    require_gpu_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--erased-embeddings", type=Path, required=True)
    parser.add_argument("--gpu-type", choices=tuple(GPU_PROFILES), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()

    profile = require_gpu_profile(args.gpu_type)
    model, tokenizer = load_lment_with_erased_embeddings(
        args.base_model,
        args.erased_embeddings,
        device="cuda",
        dtype=pick_dtype(profile.dtype),
    )
    try:
        result = evaluate_lment_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=[args.prompt],
            gpu_type=args.gpu_type,
            relevance_callback=lambda _prompt: "Fake infrastructure score\nRating: [[2]]",
            fluency_callback=lambda _prompt: "Fake infrastructure score\nRating: [[2]]",
            max_new_tokens=args.max_new_tokens,
        )
        result["judge"] = "deterministic_fake_callbacks"
        result["scientific_quality_claim"] = False
        result["embedding_device"] = str(model.get_input_embeddings().weight.device)
        if result["embedding_device"] != "cuda:0":
            raise AssertionError(f"Erased model is not on cuda:0: {result['embedding_device']}")
        if result["n"] != 1:
            raise AssertionError("Expected exactly one GPU smoke prompt")
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
