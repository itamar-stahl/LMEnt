import tempfile
import unittest
from pathlib import Path

import yaml

from ember.real_flow_test import verify_real_flow_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RealFlowContractTests(unittest.TestCase):
    def test_rejects_run_without_real_retained_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "outputs").mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "report.json"):
                verify_real_flow_run(run_dir)

    def test_windows_and_slurm_configs_define_equivalent_cpu_gpu_flows(self) -> None:
        names = {
            "windows_cpu": "ember_lment_real_windows_cpu.yaml",
            "windows_gpu": "ember_lment_real_windows_gpu.yaml",
            "slurm_cpu": "ember_lment_real_slurm_cpu.yaml",
            "slurm_gpu": "ember_lment_real_slurm_gpu.yaml",
        }
        configs = {
            key: yaml.safe_load((PROJECT_ROOT / "configs" / filename).read_text())
            for key, filename in names.items()
        }
        for key, config in configs.items():
            self.assertEqual(config["rank"], 2, key)
            self.assertEqual(config["selection"]["mode"], "threshold", key)
            self.assertEqual(config["eval"]["data_json"],
                             "../data/mc_questions_real_flow.json", key)
            self.assertFalse(config["lment"]["features"]["reuse"], key)
            self.assertEqual(config["lment"]["features"]["max_iterations"], 2, key)
            self.assertFalse(config["lment"]["save"]["full_model"], key)
        for platform in ("windows", "slurm"):
            self.assertEqual(
                configs[f"{platform}_cpu"]["lment"]["model_device"], "cpu")
            self.assertEqual(
                configs[f"{platform}_cpu"]["lment"]["features"]["fitting_device"],
                "cpu",
            )
            self.assertEqual(
                configs[f"{platform}_gpu"]["lment"]["model_device"], "cuda")
            self.assertEqual(
                configs[f"{platform}_gpu"]["lment"]["features"]["fitting_device"],
                "cuda",
            )


if __name__ == "__main__":
    unittest.main()
