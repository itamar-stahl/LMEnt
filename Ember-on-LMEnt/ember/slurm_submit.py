#!/usr/bin/env python3
"""Prepare and submit one complete LMEnt EMBER run to Slurm."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from ember.lment_pipeline import load_lment_config
from ember.lment_runs import PreparedRun, effective_config_dict, prepare_run
from ember.slurm import SlurmResources, materialize_slurm_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if not args.concept.strip():
        raise ValueError("--concept must be non-empty")
    return args


def resolve_judge_model(
    source: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local judge or download it to shared storage on the client."""
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


def prepare_submission(args: argparse.Namespace) -> Dict[str, Path]:
    """Create one complete run folder and its literal Slurm job."""
    source = load_lment_config(args.config)
    if source.slurm is None:
        raise ValueError("Slurm submission requires a lment.slurm YAML section")
    prepared = prepare_run(args.config, args.concept, execution="slurm")
    runtime = prepared.runtime_config

    if runtime.judge_model is not None:
        judge_path = resolve_judge_model(
            runtime.judge_model,
            revision=runtime.judge_revision,
            cache_dir=runtime.judge_cache_dir,
            local_files_only=runtime.judge_local_files_only,
        )
        runtime = replace(
            runtime,
            judge_model=str(judge_path),
            judge_local_files_only=True,
        )
    if runtime.alpaca_eval:
        hf_hub_download(
            repo_id="tatsu-lab/alpaca_eval",
            repo_type="dataset",
            filename="alpaca_eval.json",
            cache_dir=(
                str(runtime.judge_cache_dir.resolve())
                if runtime.judge_cache_dir is not None else None),
        )
    prepared.effective_config.write_text(
        yaml.safe_dump(effective_config_dict(runtime), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    resources = SlurmResources.from_mapping(source.slurm)
    job = materialize_slurm_job(prepared.run_dir, resources)
    with (prepared.run_dir / "client.log").open("a", encoding="utf-8") as log:
        log.write(f"client_host={platform.node()}\n")
        log.write(f"job_slurm={job.resolve()}\n")
    return {
        "run_dir": prepared.run_dir,
        "source_config": prepared.source_config,
        "config": prepared.effective_config,
        "run_wrapper": prepared.run_dir / "run_wrapper.sh",
        "run_wrapper_ps1": prepared.run_dir / "run_wrapper.ps1",
        "job_slurm": job,
    }


def submit_job(job_path: Path) -> str:
    completed = subprocess.run(
        ["sbatch", "--parsable", str(Path(job_path).resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().split(";", 1)[0]


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    written = prepare_submission(args)
    job_id = submit_job(written["job_slurm"])
    result = {
        "schema_version": 1,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "concept": args.concept,
        "run_dir": str(written["run_dir"]),
        "job_slurm": str(written["job_slurm"]),
        "job_id": job_id,
    }
    report_path = written["run_dir"] / "client_report.json"
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (written["run_dir"] / "client.log").open("a", encoding="utf-8") as log:
        log.write(f"submitted_job_id={job_id}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "build_parser", "parse_args", "prepare_submission", "resolve_judge_model",
    "submit_job", "main",
]
