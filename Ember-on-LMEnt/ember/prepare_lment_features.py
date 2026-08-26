#!/usr/bin/env python3
"""Prepare LMEnt EMBER embedding factors on a CPU login node."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from ember.lment_pipeline import ensure_factor_artifact, load_lment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--concept-json", type=Path, required=True)
    parser.add_argument("--neutral-json", type=Path, required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    concept_json = args.concept_json.resolve()
    neutral_json = args.neutral_json.resolve()
    if not concept_json.is_file():
        raise FileNotFoundError(f"Concept JSON not found: {concept_json}")
    if not neutral_json.is_file():
        raise FileNotFoundError(f"Neutral JSON not found: {neutral_json}")
    config = replace(
        load_lment_config(args.config),
        prepare_features=True,
        concept_json=concept_json,
        neutral_json=neutral_json,
        device="cpu",
    )
    ensure_factor_artifact(config, args.concept)
    print(
        f"[prepared] {args.concept}: CPU factors are ready under "
        f"{config.features_root}"
    )


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main"]
