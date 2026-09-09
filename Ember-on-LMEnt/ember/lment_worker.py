"""Execute one already-prepared LMEnt EMBER run."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from ember.erasure import io
from ember.gemma_judge import GemmaJudge
from ember.lment_pipeline import load_lment_config, run_lment_pipeline
from ember.lment_runs import publish_run_features, write_run_environment
from ember.subprocess_judge import SubprocessJudge, resolve_judge_directory


def build_judge(config) -> object | None:
    """Load the configured judge in this process or in its own interpreter.

    ``subprocess`` exists because EMBER pins transformers 4.56.2 while
    ``gemma-4-12B-it`` needs 5.x to be recognised at all. Both branches expose
    the same four seams, so the pipeline is unaffected by the choice.
    """
    if config.judge_model is None:
        return None
    if config.judge_executor == "inproc":
        return GemmaJudge.from_pretrained(
            config.judge_model,
            device=config.judge_device,
            max_new_tokens=config.judge_max_new_tokens,
            local_files_only=config.judge_local_files_only,
            cache_dir=config.judge_cache_dir,
        )
    # Resolve the pinned snapshot here so the worker never needs hub access
    # and the run snapshot records exactly the weights that were loaded.
    model_path = resolve_judge_directory(
        config.judge_model,
        revision=config.judge_revision,
        cache_dir=config.judge_cache_dir,
        local_files_only=config.judge_local_files_only,
    )
    return SubprocessJudge(
        model_path=model_path,
        python_executable=config.judge_python,
        device=config.judge_device,
        max_new_tokens=config.judge_max_new_tokens,
        cache_dir=config.judge_cache_dir,
        local_files_only=config.judge_local_files_only,
        startup_timeout=config.judge_startup_timeout,
    )


def execute_prepared_run(config_path: Path, concept: str, *,
                         execution: str) -> dict:
    config = load_lment_config(config_path)
    if config.output_dir is None:
        raise ValueError("Effective run config is missing lment.output_dir")
    run_dir = Path(config.output_dir).resolve().parent
    write_run_environment(run_dir, execution=execution)

    judge = build_judge(config)
    try:
        report = run_lment_pipeline(
            config,
            concept=concept,
            describe_callback=(judge.describe_feature if judge else None),
            classify_callback=(judge.classify_feature if judge else None),
            alpaca_relevance_callback=(
                judge.score_alpaca_relevance if judge else None),
            alpaca_fluency_callback=(judge.score_alpaca_fluency if judge else None),
        )
    finally:
        if judge is not None:
            judge.close()

    shared_root = config.feature_cache_root or config.features_root
    cache_status = publish_run_features(
        config.features_root, shared_root, config, concept)
    report["feature_cache"] = {
        "status": cache_status,
        "shared_root": str(Path(shared_root).resolve()),
        "run_root": str(Path(config.features_root).resolve()),
    }
    io.save_json_atomic(Path(config.output_dir) / "report.json", report)
    saved_path = (
        report["checkpoint_path"]
        if report["save"]["mode"] == "full_model"
        else report["erased_embeddings_path"]
    )
    print(f"[done] {concept}: delta={report['chosen_delta']} -> {saved_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument(
        "--execution", choices=("local", "windows", "slurm"), required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    execute_prepared_run(args.config, args.concept, execution=args.execution)


if __name__ == "__main__":
    main()


__all__ = ["build_judge", "build_parser", "execute_prepared_run", "main"]
