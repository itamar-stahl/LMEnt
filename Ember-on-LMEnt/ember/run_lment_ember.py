#!/usr/bin/env python3
"""Erase one or more concepts from a local LMEnt checkpoint with EMBER alone."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from ember.lment_pipeline import ensure_feature_artifacts, load_lment_config, run_concept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply standalone EMBER to a local LMEnt checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concepts", nargs="+", required=True)
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Override grid selection with an explicit embedding-edit delta.",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = load_lment_config(args.config)
    if args.delta is not None:
        config = replace(config, explicit_delta=float(args.delta))

    ensure_feature_artifacts(config, args.concepts)
    for concept in args.concepts:
        report = run_concept(config, concept=concept)
        print(
            f"[done] {concept}: delta={report['chosen_delta']} -> "
            f"{report['checkpoint_path']}")


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "parse_args", "main"]
