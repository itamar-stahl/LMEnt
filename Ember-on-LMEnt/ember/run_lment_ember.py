#!/usr/bin/env python3
"""Erase one named concept from a local LMEnt checkpoint with EMBER alone."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from ember.evals.lment_alpaca import GPU_PROFILES
from ember.gemma_judge import GemmaJudge
from ember.lment_pipeline import load_lment_config, run_lment_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply standalone EMBER to a local LMEnt checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--concept-json", type=Path, required=True)
    parser.add_argument("--neutral-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Override grid selection with an explicit embedding-edit delta.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Concept QA/Simdom train+test JSON for automatic best-delta selection.",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Select features only by ratio_abs instead of callback judging.",
    )
    parser.add_argument(
        "--feature-ratio-threshold",
        type=float,
        default=None,
        metavar="P",
        help="Required with --skip-llm-judge; keep features with ratio_abs >= P.",
    )
    parser.add_argument(
        "--full-save",
        action="store_true",
        help="Save a full Hugging Face checkpoint instead of only the erased embedding.",
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Require existing factor files; never rebuild them in this process.",
    )
    parser.add_argument("--alpaca-eval", action="store_true")
    parser.add_argument("--alpaca-split", choices=("train", "test"), default="test")
    parser.add_argument("--gpu-type", choices=tuple(GPU_PROFILES), default=None)
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Gemma instruction-model path or Hub ID used for all judge callbacks.",
    )
    parser.add_argument(
        "--judge-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--judge-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--judge-local-files-only", action="store_true",
        help="Do not contact the Hub from this process.",
    )
    parser.add_argument("--judge-cache-dir", type=Path, default=None)
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.skip_llm_judge and args.feature_ratio_threshold is None:
        parser.error("--feature-ratio-threshold P is required with --skip-llm-judge")
    if not args.skip_llm_judge and args.feature_ratio_threshold is not None:
        parser.error("--feature-ratio-threshold is only valid with --skip-llm-judge")
    if args.alpaca_eval and args.gpu_type is None:
        parser.error("--gpu-type is required with --alpaca-eval")
    if args.judge_max_new_tokens <= 0:
        parser.error("--judge-max-new-tokens must be positive")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = load_lment_config(args.config)
    config = replace(
        config,
        concept_json=args.concept_json.resolve(),
        neutral_json=args.neutral_json.resolve(),
        output_dir=args.output_dir.resolve(),
        prepare_features=not args.reuse_features,
        explicit_delta=(
            float(args.delta) if args.delta is not None else config.explicit_delta),
        eval_json=(args.eval_json.resolve() if args.eval_json is not None else config.eval_json),
        selection_mode=("threshold" if args.skip_llm_judge else "judge"),
        feature_ratio_threshold=args.feature_ratio_threshold,
        full_save=bool(args.full_save),
        alpaca_eval=bool(args.alpaca_eval),
        alpaca_split=args.alpaca_split,
        gpu_type=args.gpu_type,
    )
    if config.explicit_delta is None and config.eval_json is None:
        raise ValueError("Provide --eval-json for automatic delta selection or --delta")

    judge = None
    if args.judge_model is not None:
        judge = GemmaJudge.from_pretrained(
            args.judge_model,
            device=args.judge_device,
            max_new_tokens=args.judge_max_new_tokens,
            local_files_only=bool(args.judge_local_files_only),
            cache_dir=args.judge_cache_dir,
        )
    try:
        report = run_lment_pipeline(
            config,
            concept=args.concept,
            describe_callback=(judge.describe_feature if judge else None),
            classify_callback=(judge.classify_feature if judge else None),
            alpaca_relevance_callback=(
                judge.score_alpaca_relevance if judge else None),
            alpaca_fluency_callback=(
                judge.score_alpaca_fluency if judge else None),
        )
    finally:
        if judge is not None:
            judge.close()
    saved_path = (
        report["checkpoint_path"]
        if report["save"]["mode"] == "full_model"
        else report["erased_embeddings_path"]
    )
    print(f"[done] {args.concept}: delta={report['chosen_delta']} -> {saved_path}")


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "parse_args", "main"]
