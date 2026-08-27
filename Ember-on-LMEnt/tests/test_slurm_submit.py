import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from ember.slurm_submit import parse_args, prepare_submission, resolve_judge_model


class SlurmSubmitTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        concept = root / "concept.json"
        concept.write_text(json.dumps([
            {"concept": "Culture of Greece", "sentences": ["sentence"]},
        ]), encoding="utf-8")
        neutral = root / "neutral.json"
        neutral.write_text('[{"sentence":"neutral"}]', encoding="utf-8")
        activate = root / "activate_env.sh"
        activate.write_text("#!/bin/sh\n", encoding="utf-8")
        payload = {
            "method": "ember", "model_name": str(model.resolve()),
            "rank": 2, "seed": 42,
            "selection": {"mode": "judge", "ratio_thresh": 2.0},
            "ember": {"deltas": [1.0], "explicit_delta": 1.0},
            "eval": {"data_json": None, "alpaca": False},
            "lment": {
                "model_key": "lment-1b-control-2e", "model_device": "cuda",
                "dtype": "fp32", "runs_root": str((root / "runs").resolve()),
                "data": {"concept_json": str(concept.resolve()),
                         "neutral_json": str(neutral.resolve())},
                "features": {"cache_root": str((root / "cache").resolve()),
                             "reuse": False, "fitting_device": "cuda"},
                "judge": {"model": "google/gemma-3-12b-it", "revision": None,
                          "device": "cuda", "local_files_only": False,
                          "cache_dir": str((root / "hf").resolve())},
                "save": {"full_model": False},
                "execution": {"activate_script": str(activate.resolve())},
                "slurm": {"job_name": "ember", "account": "gpu-research",
                          "partition": "gpu", "constraint": "h100",
                          "time_minutes": 60, "cpu_mem_mb": 64000,
                          "cpus_per_task": 8},
            },
        }
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_client_materializes_complete_run_without_feature_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            judge = root / "judge"
            judge.mkdir()
            args = parse_args([
                "--config", str(config), "--concept", "Culture of Greece"])
            with patch(
                "ember.slurm_submit.resolve_judge_model",
                return_value=judge.resolve(),
            ) as resolve:
                written = prepare_submission(args)

            effective = yaml.safe_load(written["config"].read_text())
            wrapper = written["run_wrapper"].read_text(encoding="utf-8")
            job = written["job_slurm"].read_text(encoding="utf-8")
            files_exist = {
                name: (written["run_dir"] / name).exists()
                for name in ("source_config.yaml", "config.yaml", "run_wrapper.sh",
                             "run_wrapper.ps1", "client.log", "job.slurm")
            }

        resolve.assert_called_once()
        self.assertIn("Culture_of_Greece_lment-1b-control-2e_", written["run_dir"].name)
        self.assertEqual(effective["lment"]["judge"]["model"], str(judge.resolve()))
        self.assertTrue(effective["lment"]["judge"]["local_files_only"])
        self.assertIn("ember.slurm_gpu_preflight", wrapper)
        self.assertIn("ember.lment_worker", wrapper)
        self.assertNotIn("prepare_lment_features", wrapper)
        self.assertIn("#SBATCH --constraint=h100", job)
        for name, exists in files_exist.items():
            self.assertTrue(exists, name)

    def test_local_judge_directory_never_contacts_hub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "judge"
            model.mkdir()
            with patch("ember.slurm_submit.snapshot_download") as download:
                resolved = resolve_judge_model(str(model))
        self.assertEqual(resolved, model.resolve())
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
