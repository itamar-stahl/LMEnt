import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ember.lment_pipeline import LMEntRunConfig
from ember.prepare_lment_features import main
from ember.run_lment_ember import parse_args


class PrepareLMEntFeaturesTests(unittest.TestCase):
    def test_login_preparation_forces_cpu_and_explicit_json_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concept_json = root / "concept.json"
            neutral_json = root / "neutral.json"
            concept_json.write_text("[]", encoding="utf-8")
            neutral_json.write_text("[]", encoding="utf-8")
            base = LMEntRunConfig(
                model_path=root / "model",
                model_key="lment",
                features_root=root / "features",
                output_root=root / "outputs",
                device="cuda",
            )
            with (
                patch("ember.prepare_lment_features.load_lment_config",
                      return_value=base),
                patch("ember.prepare_lment_features.ensure_factor_artifact") as prepare,
            ):
                main([
                    "--config", str(root / "config.yaml"),
                    "--concept", "Any concept",
                    "--concept-json", str(concept_json),
                    "--neutral-json", str(neutral_json),
                ])

        prepared_config, prepared_concept = prepare.call_args.args
        self.assertEqual(prepared_concept, "Any concept")
        self.assertEqual(prepared_config.device, "cpu")
        self.assertTrue(prepared_config.prepare_features)
        self.assertEqual(prepared_config.concept_json, concept_json.resolve())
        self.assertEqual(prepared_config.neutral_json, neutral_json.resolve())

    def test_gpu_runner_can_require_prepared_features(self) -> None:
        args = parse_args([
            "--config", "config.yaml",
            "--concept", "Any concept",
            "--concept-json", "concept.json",
            "--neutral-json", "neutral.json",
            "--output-dir", "output",
            "--delta", "1.0",
            "--skip-llm-judge",
            "--feature-ratio-threshold", "8.0",
            "--reuse-features",
        ])
        self.assertTrue(args.reuse_features)


if __name__ == "__main__":
    unittest.main()
