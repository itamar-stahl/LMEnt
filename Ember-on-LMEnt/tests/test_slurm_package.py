import json
import tempfile
import unittest
from pathlib import Path

from ember.slurm import SlurmResources, materialize_h100_run


class SlurmPackageTests(unittest.TestCase):
    def test_materialized_run_is_h100_only_literal_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "Ember-on-LMEnt"
            project.mkdir()
            run_dir = root / "runs" / "job-1"
            written = materialize_h100_run(
                run_dir=run_dir,
                project_root=project,
                runner_args=[
                    "--config", "/shared/config.yaml",
                    "--concept", "Culture of Greece",
                    "--concept-json", "/shared/concept.json",
                    "--neutral-json", "/shared/neutral.json",
                    "--output-dir", "/shared/output",
                    "--delta", "0.5",
                    "--reuse-features",
                    "--judge-model", "/shared/gemma-3-12b-it",
                    "--judge-local-files-only",
                ],
                resources=SlurmResources(job_name="ember-test"),
            )

            slurm = written["job_slurm"].read_text(encoding="utf-8")
            wrapper = written["run_wrapper"].read_text(encoding="utf-8")
            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))

        self.assertIn("#SBATCH --account=gpu-research", slurm)
        self.assertIn("#SBATCH --partition=gpu-h100-killable", slurm)
        self.assertIn("#SBATCH --gpus=1", slurm)
        self.assertIn(str(written["run_wrapper"]), slurm)
        self.assertNotIn("${", slurm)
        self.assertIn(str(project / "activate_env.sh"), wrapper)
        self.assertIn("python -m ember.slurm_gpu_preflight --gpu-type h100", wrapper)
        self.assertIn("--reuse-features", wrapper)
        self.assertIn("--judge-local-files-only", wrapper)
        self.assertNotIn("${", wrapper)
        self.assertEqual(manifest["resources"]["partition"], "gpu-h100-killable")
        self.assertEqual(manifest["runner_args"][3], "Culture of Greece")


if __name__ == "__main__":
    unittest.main()
