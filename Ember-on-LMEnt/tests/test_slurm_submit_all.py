import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ember.slurm_submit_all import configured_concepts, submit_all


class SlurmSubmitAllTests(unittest.TestCase):
    def test_batch_is_thin_and_continues_after_one_submission_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text("placeholder", encoding="utf-8")

            def prepare(args):
                run = root / args.concept
                run.mkdir()
                job = run / "job.slurm"
                job.write_text("#!/bin/sh\n", encoding="utf-8")
                (run / "client.log").write_text("", encoding="utf-8")
                return {"run_dir": run, "job_slurm": job}

            with (
                patch("ember.slurm_submit_all.configured_concepts",
                      return_value=["First", "Second"]),
                patch("ember.slurm_submit_all.prepare_submission",
                      side_effect=prepare),
                patch("ember.slurm_submit_all.submit_job",
                      side_effect=["101", RuntimeError("sbatch failed")]),
            ):
                summary = submit_all(config)

            first_report = json.loads(
                (root / "First" / "client_report.json").read_text())
            second_report = json.loads(
                (root / "Second" / "client_report.json").read_text())

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["submitted"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(first_report["job_id"], "101")
        self.assertTrue(first_report["submitted"])
        self.assertFalse(second_report["submitted"])
        self.assertIn("sbatch failed", second_report["error"])

    def test_duplicate_concepts_are_rejected_before_submission(self) -> None:
        class Config:
            concept_json = None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concepts = root / "concepts.json"
            concepts.write_text(json.dumps([
                {"concept": "Same", "sentences": ["a"]},
                {"concept": "Same", "sentences": ["b"]},
            ]), encoding="utf-8")
            Config.concept_json = concepts
            with patch("ember.slurm_submit_all.load_lment_config",
                       return_value=Config()):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    configured_concepts(root / "config.yaml")


if __name__ == "__main__":
    unittest.main()
