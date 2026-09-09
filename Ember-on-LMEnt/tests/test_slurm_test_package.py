import tempfile
import unittest
from pathlib import Path

import yaml

from slurm.tests.prepare_test_job import prepare_test_job


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SlurmTestPackageTests(unittest.TestCase):
    def test_prepares_trackable_titan_xp_job_with_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            tests_dir = project / "slurm" / "tests"
            tests_dir.mkdir(parents=True)
            (project / "activate_env.sh").write_text("#!/bin/sh\n")
            (tests_dir / "node_test_runner.sh").write_text("#!/bin/sh\n")
            config = project / "test.yaml"
            config.write_text(yaml.safe_dump({
                "model_name": (
                    "/home/dcor/galbarak2/hf-models/"
                    "lment-1b-control-2e/"
                ),
                "lment": {"slurm": {
                    "job_name": "ember-lment-tests",
                    "partition": "studentkillable",
                    "constraint": "titan_xp",
                    "time_minutes": 30,
                    "cpu_mem_mb": 1000,
                    "cpus_per_task": 2,
                }},
            }), encoding="utf-8")

            package = prepare_test_job(config, project_root=project)
            test_dir = Path(package["test_dir"])
            job = Path(package["job_file"]).read_text(encoding="utf-8")
            wrapper = Path(package["wrapper"]).read_text(encoding="utf-8")

            self.assertTrue((test_dir / "source_config.yaml").is_file())
            self.assertTrue((test_dir / "client_environment.json").is_file())
            self.assertIn("#SBATCH --partition=studentkillable", job)
            self.assertIn('#SBATCH --constraint="titan_xp"', job)
            self.assertNotIn("#SBATCH --account=", job)
            self.assertIn(str(project / "activate_env.sh"), wrapper)
            self.assertIn(str(tests_dir / "node_test_runner.sh"), wrapper)
            self.assertNotIn("${", job + wrapper)

    def test_login_runner_runs_local_suite_before_submission(self) -> None:
        script = (
            PROJECT_ROOT / "slurm" / "tests" / "run_test.sh"
        ).read_text(encoding="utf-8")
        for command in (
            "unittest discover -s tests",
            "compileall -q ember tests slurm",
            "python -m pip check",
            "git ls-files \"*.sh\"",
            "python -m ember.slurm_model",
            "ember_lment_real_slurm_cpu.yaml",
            "python -m ember.real_flow_test",
            "--execution local",
        ):
            self.assertIn(command, script)
            self.assertLess(script.index(command), script.index("sbatch --parsable"))
        self.assertIn("ember_lment_real_slurm_gpu.yaml", script)
        self.assertIn("cpu_real_run", script)
        self.assertIn("  else\n    rc=$?", script)

    def test_node_runner_contains_only_gpu_tests_and_verified_real_flow(self) -> None:
        script = (
            PROJECT_ROOT / "slurm" / "tests" / "node_test_runner.sh"
        ).read_text(encoding="utf-8")
        for command in (
            "unittest tests.test_lment_gpu",
            "python -m ember.real_flow_test",
            "--concept Pornography",
            "--execution slurm",
            "real_flow_test_report.json",
            "validate_slurm_model_config",
        ):
            self.assertIn(command, script)
        self.assertNotIn("python -m ember.run_lment_ember", script)
        self.assertNotIn("verify_e2e_report", script)
        self.assertNotIn("unittest discover -s tests", script)
        self.assertNotIn("compileall", script)
        self.assertNotIn("pip check", script)
        self.assertIn("  else\n    rc=$?", script)

    def test_repository_test_yaml_is_bounded_for_titan_xp(self) -> None:
        config = yaml.safe_load((
            PROJECT_ROOT / "configs" / "ember_lment_real_slurm_gpu.yaml"
        ).read_text(encoding="utf-8"))
        self.assertEqual(config["selection"]["mode"], "threshold")
        self.assertEqual(
            config["model_name"],
            "/home/dcor/galbarak2/hf-models/lment-1b-control-2e/",
        )
        self.assertEqual(config["rank"], 2)
        self.assertFalse(config["eval"]["alpaca"])
        self.assertEqual(config["lment"]["features"]["max_iterations"], 2)
        self.assertEqual(
            config["lment"]["slurm"]["partition"], "studentkillable")
        self.assertEqual(config["lment"]["slurm"]["constraint"], "titan_xp")

        cpu = yaml.safe_load((
            PROJECT_ROOT / "configs" / "ember_lment_real_slurm_cpu.yaml"
        ).read_text(encoding="utf-8"))
        self.assertEqual(cpu["lment"]["model_device"], "cpu")
        self.assertEqual(cpu["lment"]["features"]["fitting_device"], "cpu")
        self.assertNotIn("slurm", cpu["lment"])

        production = yaml.safe_load((
            PROJECT_ROOT / "configs" / "ember_lment_slurm.yaml"
        ).read_text(encoding="utf-8"))
        self.assertEqual(
            production["model_name"],
            "/home/dcor/galbarak2/hf-models/lment-1b-control-2e/",
        )


if __name__ == "__main__":
    unittest.main()
