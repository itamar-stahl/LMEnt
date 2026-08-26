#!/usr/bin/env python3
"""Fail early unless a Slurm node provides the requested GPU and Conda env."""
from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-type", choices=("h100",), required=True)
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "lment":
        raise RuntimeError(
            "Expected the lment Conda environment; source activate_env.sh first")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the H100 job")
    actual = torch.cuda.get_device_name(0)
    if args.gpu_type == "h100" and "h100" not in actual.lower():
        raise RuntimeError(f"Expected an H100, but CUDA reports {actual!r}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    print(json.dumps({
        "conda_env": os.environ["CONDA_DEFAULT_ENV"],
        "gpu": actual,
        "free_vram_gib": round(free_bytes / 1024 ** 3, 3),
        "total_vram_gib": round(total_bytes / 1024 ** 3, 3),
    }))


if __name__ == "__main__":
    main()
