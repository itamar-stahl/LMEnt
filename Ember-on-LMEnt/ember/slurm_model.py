#!/usr/bin/env python3
"""Validate the shared LMEnt control checkpoints used by Slurm.

Two things are being prevented, and both still are:

1. **A path that cannot resolve on the cluster.** The Slurm YAML used to carry
   ``../../models_symlinks/win-lment-1b-control-2e``, a relative Windows
   development symlink pointing outside the repo. On the cluster it resolves to
   nothing, so the job queued, took a GPU, loaded, and only then failed. Naming
   an absolute cluster path and checking it here makes submission fail on the
   login node instead of burning an allocation.

2. **Erasing from the wrong twin.** An ablated twin already had the concept held
   out of training, so running an erasure method against it measures nothing.
   The ablated checkpoints stay rejected, and the tests pin that.

Until 2026-09-06 this was a single hard-coded path, because the cluster held one
control checkpoint. It now holds two, one per twin pair, and they are NOT
interchangeable: ``lment-1b-control-2e`` belongs to the retired Pornography pair
(batch 32,768, lr/wd from a different source) and ``lment-1b-control-2e-b131k``
to the Ancient Rome pair (batch 131,072, job 853707). Nothing trained in one
pair is weight-comparable to the other -- see ``Untaught/COMPARABILITY.md``.

A twin is only meaningful against its own control, so both are allowed and the
caller picks the one whose ablated twin holds out the concept being erased.
Allowing a second path does not weaken the check: the existence test below is
what catches an unresolvable path, and it is unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence

import yaml


SLURM_LMENT_MODEL_PATHS = (
    # Pornography pair's control; its twin held out Q291.
    "/home/dcor/galbarak2/hf-models/lment-1b-control-2e",
    # Ancient Rome pair's control; its twin held out 56 Rome QIDs.
    "/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k",
)

# For callers that need one checkpoint rather than the allowed set -- the Slurm
# test package builds its probe job from this.
SLURM_LMENT_MODEL_PATH = SLURM_LMENT_MODEL_PATHS[0]


def validate_slurm_model_config(
    config_path: Path,
    *,
    require_exists: bool = True,
) -> Path:
    """Require one of the shared cluster control checkpoints in a Slurm YAML."""
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw.get("model_name"):
        raise ValueError(f"Slurm YAML has no model_name: {config_path}")
    configured = str(PurePosixPath(str(raw["model_name"])))
    if configured not in SLURM_LMENT_MODEL_PATHS:
        allowed = ", ".join(f"{path}/" for path in SLURM_LMENT_MODEL_PATHS)
        raise ValueError(
            "Slurm runs must use a shared LMEnt control checkpoint "
            f"({allowed}); configured={raw['model_name']!r}"
        )
    model_path = Path(configured)
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


__all__ = [
    "SLURM_LMENT_MODEL_PATH",
    "SLURM_LMENT_MODEL_PATHS",
    "validate_slurm_model_config",
    "main",
]
