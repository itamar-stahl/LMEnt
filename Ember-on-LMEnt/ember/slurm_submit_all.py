#!/usr/bin/env python3
"""Submit one independent LMEnt EMBER Slurm job for every configured concept."""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ember.lment_pipeline import load_lment_config
from ember.slurm_submit import prepare_submission, submit_job, write_client_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def configured_concepts(config_path: Path) -> List[str]:
    config = load_lment_config(config_path)
    if config.concept_json is None:
        raise ValueError("YAML has no lment.data.concept_json")
    payload = json.loads(Path(config.concept_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Concept JSON must contain a list")
    concepts: List[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not str(item.get("concept", "")).strip():
            raise ValueError(f"Invalid concept record at index {index}")
        concepts.append(str(item["concept"]).strip())
    duplicates = sorted({name for name in concepts if concepts.count(name) > 1})
    if duplicates:
        raise ValueError(f"Concept JSON contains duplicate concepts: {duplicates}")
    if not concepts:
        raise ValueError("Concept JSON contains no concepts")
    return concepts


def submit_all(config_path: Path) -> Dict[str, object]:
    """Submit every concept independently and continue after submission errors."""
    config_path = Path(config_path).resolve()
    entries: List[Dict[str, object]] = []
    for concept in configured_concepts(config_path):
        written = None
        try:
            written = prepare_submission(Namespace(
                config=config_path,
                concept=concept,
            ))
            job_id = submit_job(written["job_slurm"])
            result = write_client_report(written, concept, job_id=job_id)
            entry = {
                "concept": concept,
                "status": "submitted",
                "run_dir": result["run_dir"],
                "job_id": job_id,
                "error": None,
            }
            print(
                f"[submitted] {concept}: job={job_id} run={result['run_dir']}",
                file=sys.stderr,
            )
        except Exception as error:
            if written is not None:
                result = write_client_report(
                    written, concept, job_id=None, error=str(error))
                run_dir = result["run_dir"]
            else:
                run_dir = None
            entry = {
                "concept": concept,
                "status": "failed",
                "run_dir": run_dir,
                "job_id": None,
                "error": str(error),
            }
            print(f"[failed] {concept}: {error}", file=sys.stderr)
        entries.append(entry)
    failed = sum(entry["status"] == "failed" for entry in entries)
    return {
        "schema_version": 1,
        "config": str(config_path),
        "total": len(entries),
        "submitted": len(entries) - failed,
        "failed": failed,
        "entries": entries,
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = submit_all(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "configured_concepts", "submit_all", "main"]
