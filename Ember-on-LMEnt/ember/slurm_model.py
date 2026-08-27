#!/usr/bin/env python3
"""Validate the one shared LMEnt control checkpoint used by Slurm."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence

import yaml


SLURM_LMENT_MODEL_PATH = (
    "/home/dcor/galbarak2/hf-models/lment-1b-control-2e"
)


def validate_slurm_model_config(
    config_path: Path,
    *,
    require_exists: bool = True,
) -> Path:
    """Require the exact shared cluster checkpoint in a Slurm YAML."""
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw.get("model_name"):
        raise ValueError(f"Slurm YAML has no model_name: {config_path}")
    configured = str(PurePosixPath(str(raw["model_name"])))
    if configured != SLURM_LMENT_MODEL_PATH:
        raise ValueError(
            "Slurm runs must use the shared LMEnt control checkpoint: "
            f"{SLURM_LMENT_MODEL_PATH}/; configured={raw['model_name']!r}"
        )
    model_path = Path(SLURM_LMENT_MODEL_PATH)
    if require_exists and not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            "Shared LMEnt control checkpoint is unavailable or incomplete: "
            f"{model_path / 'config.json'}"
        )
    return model_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(validate_slurm_model_config(args.config))


if __name__ == "__main__":
    main()


__all__ = ["SLURM_LMENT_MODEL_PATH", "validate_slurm_model_config", "main"]
