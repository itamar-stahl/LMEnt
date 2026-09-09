import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ember.lment_pipeline import LMEntRunConfig, ensure_factor_artifact, load_lment_config
from ember.run_lment_ember import parse_args
from ember.train_mf_features import parse_args as parse_feature_args


def write_config(path: Path) -> None:
    path.write_text(
        """method: ember
model_name: ../model
rank: 7
seed: 9
selection:
  mode: threshold
  ratio_thresh: 2.5
  feature_ratio_threshold: 8.0
ember:
  deltas: [0.5, 1.0]
  explicit_delta: 1.0
eval:
  data_json: ../eval.json
lment:
  model_key: local-model
  model_device: cpu
  dtype: fp32
  runs_root: ../runs
  data:
    concept_json: ../concept.json
    neutral_json: ../neutral.json
  features:
    cache_root: ../features
    reuse: false
    fitting_device: cuda
  judge:
    model: null
  save:
    full_model: false
  execution:
    activate_script: ../activate_env.sh
""",
        encoding="utf-8",
    )


class LMEntConfigTests(unittest.TestCase):
    def test_standalone_feature_fitting_defaults_to_cuda(self) -> None:
        with patch("sys.argv", ["train_mf_features.py"]):
            args = parse_feature_args()
        self.assertEqual(args.fitting_device, "cuda")

    def test_public_cli_only_accepts_config_and_concept(self) -> None:
        args = parse_args(["--config", "config.yaml", "--concept", "Concept A"])
        self.assertEqual(args.concept, "Concept A")
        self.assertEqual(args.config, Path("config.yaml"))

    def test_judge_executor_defaults_to_in_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            write_config(path)
            config = load_lment_config(path)
        self.assertEqual(config.judge_executor, "inproc")
        self.assertIsNone(config.judge_python)

    def test_subprocess_executor_requires_an_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            write_config(path)
            path.write_text(path.read_text().replace(
                "  judge:\n    model: null",
                "  judge:\n    model: google/gemma-4-12B-it\n"
                "    executor: subprocess"), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_lment_config(path)
        self.assertIn("lment.judge.python is required", str(caught.exception))

    def test_unknown_judge_executor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            write_config(path)
            path.write_text(path.read_text().replace(
                "  judge:\n    model: null",
                "  judge:\n    model: null\n    executor: http"), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_lment_config(path)
        self.assertIn("executor must be", str(caught.exception))

    def test_subprocess_executor_accepts_a_configured_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            write_config(path)
            interpreter = Path(tmp) / "python"
            interpreter.write_text("", encoding="utf-8")
            path.write_text(path.read_text().replace(
                "  judge:\n    model: null",
                "  judge:\n    model: google/gemma-4-12B-it\n"
                "    executor: subprocess\n    python: ./python"), encoding="utf-8")
            config = load_lment_config(path)
        self.assertEqual(config.judge_executor, "subprocess")
        self.assertEqual(config.judge_python, interpreter.resolve())


    def test_judge_startup_timeout_defaults_to_an_hour(self) -> None:
        # The lab filers read at ~25 MB/s, so the default has to outlast a
        # 22 GiB load or a healthy judge reads as a dead one.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            write_config(path)
            config = load_lment_config(path)
        self.assertEqual(config.judge_startup_timeout, 3600.0)

    def test_non_positive_judge_startup_timeout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            write_config(path)
            path.write_text(path.read_text().replace(
                "  judge:\n    model: null",
                "  judge:\n    model: null\n    startup_timeout_seconds: 0"),
                encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_lment_config(path)
        self.assertIn("startup_timeout_seconds must be positive",
                      str(caught.exception))

    def test_yaml_matches_original_ember_shape_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            config_dir.mkdir()
            path = config_dir / "lment.yaml"
            write_config(path)
            config = load_lment_config(path)

        self.assertEqual(config.model_path, root / "model")
        self.assertEqual(config.features_root, root / "features")
        self.assertEqual(config.feature_cache_root, root / "features")
        self.assertEqual(config.runs_root, root / "runs")
        self.assertEqual(config.model_key, "local-model")
        self.assertEqual(config.rank, 7)
        self.assertEqual(list(config.deltas), [0.5, 1.0])
        self.assertEqual(config.fitting_device, "cuda")

    def test_feature_fitting_uses_configured_cuda_device(self) -> None:
        config = LMEntRunConfig(
            model_path=Path("model"),
            model_key="local-model",
            features_root=Path("features"),
            runs_root=Path("runs"),
            fitting_device="cuda",
            concept_json=Path("concept.json"),
            neutral_json=Path("neutral.json"),
            selection_mode="threshold",
            feature_ratio_threshold=2.0,
        )
        fake_paths = tuple(Path(name) for name in (
            "missing-artifact.pkl", "missing-stats.csv", "missing-tokens.csv"))

        def create_outputs(*_args, **_kwargs):
            for output in fake_paths:
                output.touch()

        with (
            patch("ember.lment_pipeline._embedding_training_paths", return_value=fake_paths),
            patch("ember.lment_pipeline.torch.cuda.is_available", return_value=True),
            patch("ember.lment_pipeline.subprocess.run", side_effect=create_outputs) as run,
        ):
            try:
                ensure_factor_artifact(config, "Any concept")
            finally:
                for output in fake_paths:
                    output.unlink(missing_ok=True)

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--fitting-device") + 1], "cuda")

    def test_cuda_feature_fitting_fails_instead_of_falling_back(self) -> None:
        config = LMEntRunConfig(
            model_path=Path("model"), model_key="model",
            features_root=Path("features"), runs_root=Path("runs"),
            fitting_device="cuda", concept_json=Path("concept.json"),
            neutral_json=Path("neutral.json"), selection_mode="threshold",
            feature_ratio_threshold=2.0,
        )
        with (
            patch("ember.lment_pipeline._embedding_training_paths", return_value=(
                Path("missing-a"), Path("missing-b"), Path("missing-c"))),
            patch("ember.lment_pipeline.torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "fitting_device: cpu"):
                ensure_factor_artifact(config, "Concept")


if __name__ == "__main__":
    unittest.main()
