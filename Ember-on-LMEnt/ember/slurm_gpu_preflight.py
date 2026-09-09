#!/usr/bin/env python3
"""Fail early unless a Slurm node provides CUDA in the LMEnt environment."""
from __future__ import annotations

import json
import os

import torch


def main() -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "lment":
        raise RuntimeError(
            "Expected the lment Conda environment; source activate_env.sh first")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Slurm job")
    actual = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    print(json.dumps({
        "conda_env": os.environ["CONDA_DEFAULT_ENV"],
        "gpu": actual,
        "free_vram_gib": round(free_bytes / 1024 ** 3, 3),
        "total_vram_gib": round(total_bytes / 1024 ** 3, 3),
    }))


if __name__ == "__main__":
    main()
