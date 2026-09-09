#!/usr/bin/env python3
"""Create one trackable Slurm job that runs the complete test package."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import sys
from typing import Any, Dict, Optional, Sequence

import yaml

from ember.slurm import SlurmResources, materialize_slurm_job
from ember.slurm_model import (
    SLURM_LMENT_MODEL_PATH,
    validate_slurm_model_config,
)


def _allocate_test_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stem = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{stem}_{suffix}"
    candidate.mkdir()
    return candidate.resolve()


def prepare_test_job(config_path: Path, *,
                     project_root: Optional[Path] = None) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Slurm test config not found: {config_path}")
    project = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    activate = project / "activate_env.sh"
    node_runner = project / "slurm" / "tests" / "node_test_runner.sh"
    for label, path in (("activation script", activate),
                        ("node test runner", node_runner)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_slurm_model_config(config_path, require_exists=False)
    try:
        slurm_payload = raw["lment"]["slurm"]
    except (KeyError, TypeError) as error:
        raise ValueError("Test YAML must contain lment.slurm settings") from error
    resources = SlurmResources.from_mapping(slurm_payload)
    test_dir = _allocate_test_dir(project / "slurm_test_runs")
    copied_config = test_dir / "source_config.yaml"
    shutil.copy2(config_path, copied_config)

    command = [
        "sh", str(node_runner.resolve()),
        "--project", str(project),
        "--config", str(config_path),
        "--test-dir", str(test_dir),
    ]
    wrapper = test_dir / "run_wrapper.sh"
    wrapper.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f". {shlex.quote(str(activate.resolve()))}\n"
        f"exec {shlex.join(command)}\n",
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IXUSR)
    job = materialize_slurm_job(test_dir, resources)
    environment = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(project),
        "source_config": str(config_path),
        "copied_config": str(copied_config),
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "hostname": platform.node(),
        "slurm": dict(slurm_payload),
        "cluster_model_path": SLURM_LMENT_MODEL_PATH,
    }
    environment_path = test_dir / "client_environment.json"
    environment_path.write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "test_dir": str(test_dir),
        "job_file": str(job.resolve()),
        "wrapper": str(wrapper.resolve()),
        "source_config": str(copied_config.resolve()),
        "client_environment": str(environment_path.resolve()),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(
        prepare_test_job(args.config, project_root=args.project), indent=2))


if __name__ == "__main__":
    main()


__all__ = ["prepare_test_job", "main"]
