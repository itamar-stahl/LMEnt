import tempfile
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNMF_ROOT = PROJECT_ROOT / "external" / "snmf"
if str(SNMF_ROOT) not in sys.path:
    sys.path.insert(0, str(SNMF_ROOT))

from ember.lment_pipeline import load_lment_config
from ember.run_lment_ember import parse_args


class LMEntConfigTests(unittest.TestCase):
    def test_cli_accepts_one_or_more_general_concepts(self) -> None:
        args = parse_args([
            "--config", "config.yaml",
            "--concepts", "Concept A", "Concept B",
            "--delta", "5.0",
        ])
        self.assertEqual(args.concepts, ["Concept A", "Concept B"])
        self.assertEqual(args.delta, 5.0)

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


if __name__ == "__main__":
    unittest.main()
