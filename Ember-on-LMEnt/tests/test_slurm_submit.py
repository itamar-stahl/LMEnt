import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ember.lment_pipeline import LMEntRunConfig
from ember.slurm_submit import parse_args, prepare_submission, resolve_judge_model


class SlurmSubmitTests(unittest.TestCase):
    def test_login_flow_prepares_cpu_then_materializes_cuda_judge_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "Ember-on-LMEnt"
            project.mkdir()
            concept_json = root / "concept.json"
            neutral_json = root / "neutral.json"
            config_path = root / "config.yaml"
            output_dir = root / "erased"
            judge_path = root / "gemma-3-12b-it"
            for path in (concept_json, neutral_json, config_path):
                path.write_text("{}", encoding="utf-8")
            judge_path.mkdir()
            hf_home = root / "hf-home"
            config = LMEntRunConfig(
                model_path=root / "lment-model",
                model_key="lment",
                features_root=root / "features",
                output_root=root / "outputs",
                device="cuda",
            )
            args = parse_args([
                "--config", str(config_path),
                "--concept", "Culture of Greece",
                "--concept-json", str(concept_json),
                "--neutral-json", str(neutral_json),
                "--output-dir", str(output_dir),
                "--delta", "0.5",
                "--judge-model", "google/gemma-3-12b-it",
                "--alpaca-eval",
                "--alpaca-max-items", "1",
                "--hf-home", str(hf_home),
                "--run-root", str(root / "runs"),
                "--dry-run",
            ])
            with (
                patch("ember.slurm_submit.load_lment_config", return_value=config),
                patch("ember.slurm_submit.ensure_factor_artifact") as prepare,
                patch("ember.slurm_submit.resolve_judge_model",
                      return_value=judge_path.resolve()) as resolve,
                patch("ember.slurm_submit.hf_hub_download") as stage_alpaca,
            ):
                written = prepare_submission(args, project_root=project)

            prepared_config, prepared_concept = prepare.call_args.args
            manifest = json.loads(
                written["manifest"].read_text(encoding="utf-8"))

        self.assertEqual(prepared_concept, "Culture of Greece")
        self.assertEqual(prepared_config.device, "cpu")
        self.assertTrue(prepared_config.prepare_features)
        resolve.assert_called_once()
        runner_args = manifest["runner_args"]
        self.assertIn("--reuse-features", runner_args)
        self.assertEqual(
            runner_args[runner_args.index("--model-device") + 1], "cuda")
        self.assertEqual(
            runner_args[runner_args.index("--judge-model") + 1], str(judge_path.resolve()))
        self.assertIn("--judge-local-files-only", runner_args)
        self.assertEqual(
            runner_args[runner_args.index("--alpaca-max-items") + 1], "1")
        self.assertEqual(manifest["environment"]["HF_HOME"], str(hf_home.resolve()))
        stage_alpaca.assert_called_once()
        self.assertFalse(output_dir.exists())

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
