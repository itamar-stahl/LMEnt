"""Materialize an auditable Slurm job inside a prepared LMEnt run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class SlurmResources:
    job_name: str
    account: str | None
    partition: str
    constraint: str
    time_minutes: int
    cpu_mem_mb: int
    cpus_per_task: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SlurmResources":
        required = {
            "job_name", "partition", "constraint",
            "time_minutes", "cpu_mem_mb", "cpus_per_task",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Missing lment.slurm keys: {sorted(missing)}")
        allowed = required | {"account"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown lment.slurm keys: {sorted(unknown)}")
        resources = cls(
            job_name=str(payload["job_name"]),
            account=(str(payload["account"]) if payload.get("account") else None),
            partition=str(payload["partition"]),
            constraint=str(payload["constraint"]),
            time_minutes=int(payload["time_minutes"]),
            cpu_mem_mb=int(payload["cpu_mem_mb"]),
            cpus_per_task=int(payload["cpus_per_task"]),
        )
        resources.validate()
        return resources

    def validate(self) -> None:
        for name in ("job_name", "partition"):
            if not getattr(self, name).strip():
                raise ValueError(f"SLURM {name} must be non-empty")
        if self.account is not None and not self.account.strip():
            raise ValueError("SLURM account must be non-empty when provided")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.constraint):
            raise ValueError(
                "SLURM constraint must contain only letters, numbers, '_', '-', or '.'"
            )
        if min(self.time_minutes, self.cpu_mem_mb, self.cpus_per_task) <= 0:
            raise ValueError("SLURM time, memory, and CPU values must be positive")


def materialize_slurm_job(run_dir: Path,
                          resources: SlurmResources) -> Path:
    """Write ``job.slurm`` using literal absolute paths only."""
    resources.validate()
    run_dir = Path(run_dir).resolve()
    wrapper = (run_dir / "run_wrapper.sh").resolve()
    if not wrapper.is_file():
        raise FileNotFoundError(f"Prepared run wrapper not found: {wrapper}")
    job = run_dir / "job.slurm"
    account_line = (
        f"#SBATCH --account={resources.account}\n"
        if resources.account is not None else ""
    )
    text = f"""#!/bin/sh
#SBATCH --job-name={resources.job_name}
#SBATCH --output={run_dir / 'log.out'}
#SBATCH --error={run_dir / 'log.err'}
{account_line}#SBATCH --partition={resources.partition}
#SBATCH --constraint="{resources.constraint}"
#SBATCH --time={resources.time_minutes}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem={resources.cpu_mem_mb}
#SBATCH --cpus-per-task={resources.cpus_per_task}
#SBATCH --gpus=1

{wrapper}
"""
    if "${" in text:
        raise ValueError("Generated job.slurm contains an unresolved shell variable")
    job.write_text(text, encoding="utf-8", newline="\n")
    return job


__all__ = ["SlurmResources", "materialize_slurm_job"]
