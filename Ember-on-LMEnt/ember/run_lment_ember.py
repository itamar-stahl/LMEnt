#!/usr/bin/env python3
"""Run standalone EMBER erasure for one concept using one YAML config."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from ember.lment_runs import prepare_run
from ember.lment_worker import execute_prepared_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply standalone EMBER to one concept in a local LMEnt checkpoint.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if not args.concept.strip():
        raise ValueError("--concept must be non-empty")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    execution = "windows" if os.name == "nt" else "local"
    prepared = prepare_run(args.config, args.concept, execution=execution)
    execute_prepared_run(
        prepared.effective_config, args.concept, execution=execution)
    print(f"[run] {prepared.run_dir}")


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "parse_args", "main"]
