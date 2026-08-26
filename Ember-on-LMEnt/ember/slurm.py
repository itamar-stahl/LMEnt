"""Materialize auditable single-H100 jobs for the LMEnt EMBER runner."""
from __future__ import annotations

import json
import os
import re
import shlex
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SlurmResources:
    job_name: str = "ember-lment"
    account: str = "gpu-research"
    partition: str = "gpu-h100-killable"
    time_minutes: int = 360
    cpu_mem_mb: int = 64_000
    cpus_per_task: int = 8

    def validate(self) -> None:
        if not self.job_name.strip():
            raise ValueError("SLURM job_name must be non-empty")
        if "h100" not in self.partition.lower():
            raise ValueError("LMEnt EMBER GPU jobs must use an H100 partition")
        if min(self.time_minutes, self.cpu_mem_mb, self.cpus_per_task) <= 0:
            raise ValueError("SLURM time, memory, and CPU values must be positive")


def materialize_h100_run(
    *,
    run_dir: str | Path,
    project_root: str | Path,
    runner_args: Sequence[str],
    resources: SlurmResources = SlurmResources(),
    environment: Optional[Mapping[str, str]] = None,
) -> Dict[str, Path]:
    """Write a literal job, wrapper, and manifest into one new run folder."""
    resources.validate()
    run_dir = Path(run_dir).resolve()
    project_root = Path(project_root).resolve()
    if run_dir.exists():
        raise FileExistsError(f"SLURM run directory already exists: {run_dir}")
    if not runner_args:
        raise ValueError("runner_args must not be empty")
    run_dir.mkdir(parents=True, exist_ok=False)

    wrapper_path = run_dir / "run_wrapper.sh"
    slurm_path = run_dir / "job.slurm"
    manifest_path = run_dir / "manifest.json"
    command = ["python", "-m", "ember.run_lment_ember", *map(str, runner_args)]
    environment = dict(environment or {})
    for key in environment:
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
    environment_lines = "".join(
        f"export {key}={shlex.quote(str(value))}\n"
        for key, value in sorted(environment.items())
    )
    wrapper = f"""#!/bin/sh
# Generated for one LMEnt EMBER H100 run. All paths and arguments are literal.
set -eu

echo "[ember] node=$(hostname) started=$(date)"
. {shlex.quote(str(project_root / 'activate_env.sh'))}
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
{environment_lines}nvidia-smi
python -m ember.slurm_gpu_preflight --gpu-type h100
{shlex.join(command)}
echo "[ember] finished=$(date)"
"""
    wrapper_path.write_text(wrapper, encoding="utf-8", newline="\n")
    os.chmod(
        wrapper_path,
        os.stat(wrapper_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    slurm = f"""#!/bin/sh
#SBATCH --job-name={resources.job_name}
#SBATCH --output={run_dir / 'log.out'}
#SBATCH --error={run_dir / 'log.err'}
#SBATCH --account={resources.account}
#SBATCH --partition={resources.partition}
#SBATCH --time={resources.time_minutes}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem={resources.cpu_mem_mb}
#SBATCH --cpus-per-task={resources.cpus_per_task}
#SBATCH --gpus=1

{wrapper_path}
"""
    slurm_path.write_text(slurm, encoding="utf-8", newline="\n")

    manifest = {
        "schema_version": 1,
        "kind": "lment_ember_h100_run",
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "resources": asdict(resources),
        "runner_args": list(map(str, runner_args)),
        "command": command,
        "environment": environment,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "run_dir": run_dir,
        "job_slurm": slurm_path,
        "run_wrapper": wrapper_path,
        "manifest": manifest_path,
    }


__all__ = ["SlurmResources", "materialize_h100_run"]
