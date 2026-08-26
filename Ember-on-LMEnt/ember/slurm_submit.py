#!/usr/bin/env python3
"""Prepare CPU factors on the login node and submit one LMEnt EMBER H100 job."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download, snapshot_download

from ember.lment_pipeline import ensure_factor_artifact, load_lment_config
from ember.slurm import SlurmResources, materialize_h100_run


DEFAULT_JUDGE_MODEL = "google/gemma-3-12b-it"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--concept-json", type=Path, required=True)
    parser.add_argument("--neutral-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--feature-ratio-threshold", type=float, default=None)
    parser.add_argument("--full-save", action="store_true")
    parser.add_argument("--alpaca-eval", action="store_true")
    parser.add_argument("--alpaca-split", choices=("train", "test"), default="test")
    parser.add_argument("--alpaca-max-items", type=int, default=None)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-revision", default=None)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--hub-local-files-only", action="store_true")
    parser.add_argument("--run-root", type=Path, default=Path("slurm_runs"))
    parser.add_argument("--job-name", default="ember-lment")
    parser.add_argument("--account", default="gpu-research")
    parser.add_argument("--partition", default="gpu-h100-killable")
    parser.add_argument("--time-minutes", type=int, default=360)
    parser.add_argument("--cpu-mem-mb", type=int, default=64_000)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Prepare and materialize the run, but do not call sbatch.",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.delta is None and args.eval_json is None:
        parser.error("provide --delta or --eval-json")
    if args.skip_llm_judge and args.feature_ratio_threshold is None:
        parser.error("--feature-ratio-threshold is required with --skip-llm-judge")
    if not args.skip_llm_judge and args.feature_ratio_threshold is not None:
        parser.error("--feature-ratio-threshold requires --skip-llm-judge")
    if args.judge_max_new_tokens <= 0:
        parser.error("--judge-max-new-tokens must be positive")
    if args.alpaca_max_items is not None and not args.alpaca_eval:
        parser.error("--alpaca-max-items requires --alpaca-eval")
    if args.alpaca_max_items is not None and args.alpaca_max_items <= 0:
        parser.error("--alpaca-max-items must be positive")
    return args


def resolve_judge_model(
    source: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local directory or cache a hosted model on shared storage."""
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.is_absolute() or source.startswith("."):
        raise FileNotFoundError(f"Judge model directory not found: {candidate}")
    resolved = snapshot_download(
        repo_id=source,
        revision=revision,
        cache_dir=(str(cache_dir.resolve()) if cache_dir is not None else None),
        local_files_only=bool(local_files_only),
    )
    return Path(resolved).resolve()


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return safe or "concept"


def _runner_args(args: argparse.Namespace, judge_path: Optional[Path]) -> List[str]:
    runner = [
        "--config", str(args.config.resolve()),
        "--model-device", "cuda",
        "--concept", args.concept,
        "--concept-json", str(args.concept_json.resolve()),
        "--neutral-json", str(args.neutral_json.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--reuse-features",
    ]
    if args.delta is not None:
        runner.extend(("--delta", str(float(args.delta))))
    if args.eval_json is not None:
        runner.extend(("--eval-json", str(args.eval_json.resolve())))
    if args.skip_llm_judge:
        runner.extend((
            "--skip-llm-judge",
            "--feature-ratio-threshold", str(float(args.feature_ratio_threshold)),
        ))
    if args.full_save:
        runner.append("--full-save")
    if args.alpaca_eval:
        runner.extend((
            "--alpaca-eval", "--alpaca-split", args.alpaca_split,
            "--gpu-type", "h100",
        ))
        if args.alpaca_max_items is not None:
            runner.extend(("--alpaca-max-items", str(args.alpaca_max_items)))
    if judge_path is not None:
        runner.extend((
            "--judge-model", str(judge_path),
            "--judge-device", "cuda",
            "--judge-max-new-tokens", str(args.judge_max_new_tokens),
            "--judge-local-files-only",
        ))
    return runner


def prepare_submission(
    args: argparse.Namespace,
    *,
    project_root: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Run the client phase and materialize, but do not submit, one H100 run."""
    project = (
        Path(project_root).resolve()
        if project_root is not None else Path(__file__).resolve().parents[1]
    )
    concept_json = args.concept_json.resolve()
    neutral_json = args.neutral_json.resolve()
    for label, path in (("concept", concept_json), ("neutral", neutral_json)):
        if not path.is_file():
            raise FileNotFoundError(f"{label.title()} JSON not found: {path}")
    if args.eval_json is not None and not args.eval_json.resolve().is_file():
        raise FileNotFoundError(f"Evaluation JSON not found: {args.eval_json.resolve()}")

    client_config = replace(
        load_lment_config(args.config),
        prepare_features=True,
        concept_json=concept_json,
        neutral_json=neutral_json,
        device="cpu",
    )
    ensure_factor_artifact(client_config, args.concept)

    needs_judge = (not args.skip_llm_judge) or args.alpaca_eval
    hf_home = args.hf_home.resolve() if args.hf_home is not None else None
    hub_cache = hf_home / "hub" if hf_home is not None else None
    judge_path = None
    if needs_judge:
        judge_path = resolve_judge_model(
            args.judge_model,
            revision=args.judge_revision,
            cache_dir=hub_cache,
            local_files_only=args.hub_local_files_only,
        )
    if args.alpaca_eval:
        hf_hub_download(
            repo_id="tatsu-lab/alpaca_eval",
            repo_type="dataset",
            filename="alpaca_eval.json",
            cache_dir=(str(hub_cache) if hub_cache is not None else None),
            local_files_only=bool(args.hub_local_files_only),
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.run_root.resolve() / f"{_safe_name(args.concept)}_{stamp}"
    resources = SlurmResources(
        job_name=args.job_name,
        account=args.account,
        partition=args.partition,
        time_minutes=args.time_minutes,
        cpu_mem_mb=args.cpu_mem_mb,
        cpus_per_task=args.cpus,
    )
    return materialize_h100_run(
        run_dir=run_dir,
        project_root=project,
        runner_args=_runner_args(args, judge_path),
        resources=resources,
        environment=({"HF_HOME": str(hf_home)} if hf_home is not None else None),
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    written = prepare_submission(args)
    result = {
        "run_dir": str(written["run_dir"]),
        "job_slurm": str(written["job_slurm"]),
        "submitted": False,
        "job_id": None,
    }
    if not args.dry_run:
        completed = subprocess.run(
            ["sbatch", "--parsable", str(written["job_slurm"])],
            check=True,
            capture_output=True,
            text=True,
        )
        result["submitted"] = True
        result["job_id"] = completed.stdout.strip().split(";", 1)[0]
    report_path = written["run_dir"] / "client_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["client_report"] = str(report_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_JUDGE_MODEL", "build_parser", "parse_args", "prepare_submission",
    "resolve_judge_model", "main",
]
