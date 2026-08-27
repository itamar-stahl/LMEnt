import tempfile
import unittest
from pathlib import Path

from ember.slurm import SlurmResources, materialize_slurm_job


class SlurmPackageTests(unittest.TestCase):
    def test_job_uses_h100_constraint_and_absolute_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp).resolve() / "run"
            run.mkdir()
            wrapper = run / "run_wrapper.sh"
            wrapper.write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
            job = materialize_slurm_job(
                run,
                SlurmResources(
                    job_name="ember", account="gpu-research",
                    partition="gpu", constraint="h100", time_minutes=60,
                    cpu_mem_mb=1000, cpus_per_task=2,
                ),
            )
            text = job.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --constraint=h100", text)
        self.assertIn(str(wrapper.resolve()), text)
        self.assertNotIn("${", text)

    def test_constraint_must_be_h100(self) -> None:
        resources = SlurmResources(
            job_name="ember", account="gpu-research", partition="gpu",
            constraint="a100", time_minutes=60, cpu_mem_mb=1000,
            cpus_per_task=2,
        )
        with self.assertRaisesRegex(ValueError, "constraint: h100"):
            resources.validate()


if __name__ == "__main__":
    unittest.main()
