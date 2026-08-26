import tempfile
import unittest
from pathlib import Path
import sys
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNMF_ROOT = PROJECT_ROOT / "external" / "snmf"
if str(SNMF_ROOT) not in sys.path:
    sys.path.insert(0, str(SNMF_ROOT))

from ember.lment_pipeline import LMEntRunConfig, ensure_factor_artifact, load_lment_config
from ember.run_lment_ember import parse_args
from tests.lment_erasure_smoke import build_parser as build_smoke_parser


class LMEntConfigTests(unittest.TestCase):
    def test_cli_requires_one_named_concept_and_explicit_threshold(self) -> None:
        args = parse_args([
            "--config", "config.yaml",
            "--concept", "Concept A",
            "--concept-json", "concept.json",
            "--neutral-json", "neutral.json",
            "--output-dir", "erased-model",
            "--delta", "5.0",
            "--skip-llm-judge",
            "--feature-ratio-threshold", "8.0",
        ])
        self.assertEqual(args.concept, "Concept A")
        self.assertEqual(args.delta, 5.0)
        self.assertEqual(args.feature_ratio_threshold, 8.0)

    def test_smoke_cli_can_preserve_erased_checkpoint(self) -> None:
        args = build_smoke_parser().parse_args([
            "--model-path", "model",
            "--concept-json", "concept.json",
            "--neutral-json", "neutral.json",
            "--keep-erased-model",
            "--output-root", "saved-smoke",
        ])
        self.assertTrue(args.keep_erased_model)
        self.assertEqual(args.output_root, Path("saved-smoke"))

    def test_cli_rejects_missing_threshold_and_gpu_profile(self) -> None:
        common = [
            "--config", "config.yaml",
            "--concept", "Concept A",
            "--concept-json", "concept.json",
            "--neutral-json", "neutral.json",
            "--output-dir", "output",
            "--delta", "1.0",
        ]
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(common + ["--skip-llm-judge"])
            with self.assertRaises(SystemExit):
                parse_args(common + ["--alpaca-eval"])

    def test_yaml_paths_resolve_relative_to_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            path = config_dir / "lment.yaml"
            path.write_text(
                "\n".join([
                    "model_path: ../model",
                    "model_key: local-model",
                    "features_root: ../features",
                    "output_root: ../outputs",
                    "eval_json: ../eval.json",
                    "rank: 7",
                    "seed: 9",
                    "ratio_thresh: 2.5",
                    "deltas: [0.5, 1.0]",
                    "device: cpu",
                    "dtype: fp32",
                ]),
                encoding="utf-8",
            )

            config = load_lment_config(path)

        self.assertEqual(config.model_path, root / "model")
        self.assertEqual(config.features_root, root / "features")
        self.assertEqual(config.output_root, root / "outputs")
        self.assertEqual(config.eval_json, root / "eval.json")
        self.assertEqual(config.model_key, "local-model")
        self.assertEqual(config.rank, 7)
        self.assertEqual(list(config.deltas), [0.5, 1.0])

    def test_feature_preparation_forces_cpu_factorization(self) -> None:
        config = LMEntRunConfig(
            model_path=Path("model"),
            model_key="local-model",
            features_root=Path("features"),
            output_root=Path("outputs"),
            prepare_features=True,
            concept_json=Path("concept.json"),
            neutral_json=Path("neutral.json"),
        )
        fake_paths = (
            Path("missing-artifact.pkl"),
            Path("missing-stats.csv"),
            Path("missing-tokens.csv"),
        )

        def create_outputs(*_args, **_kwargs):
            for path in fake_paths:
                path.touch()

        with (
            patch("ember.lment_pipeline._embedding_training_paths", return_value=fake_paths),
            patch("ember.lment_pipeline.subprocess.run", side_effect=create_outputs) as run,
        ):
            try:
                ensure_factor_artifact(config, "Any concept")
            finally:
                for path in fake_paths:
                    path.unlink(missing_ok=True)

        factor_command = run.call_args.args[0]
        fitting_index = factor_command.index("--fitting-device")
        self.assertEqual(factor_command[fitting_index + 1], "cpu")


if __name__ == "__main__":
    unittest.main()
