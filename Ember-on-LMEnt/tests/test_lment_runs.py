import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml

from ember.lment_runs import (
    FEATURE_BRANCHES,
    allocate_run_dir,
    feature_concept_dirs,
    prepare_run,
    publish_run_features,
    validate_feature_bundle,
)


class LMEntRunFolderTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"olmo2"}\n')
        concept = root / "concept.json"
        concept.write_text(json.dumps([
            {"concept": "Other", "sentences": ["other"]},
            {"concept": "Target", "sentences": ["first", "second"]},
        ]), encoding="utf-8")
        neutral = root / "neutral.json"
        neutral.write_text(json.dumps([
            {"sentence": "neutral one"}, {"sentence": "neutral two"},
        ]), encoding="utf-8")
        config = {
            "method": "ember",
            "model_name": str(model.resolve()),
            "rank": 2,
            "seed": 3,
            "selection": {
                "mode": "threshold", "ratio_thresh": 2.0,
                "feature_ratio_threshold": 4.0,
            },
            "ember": {"deltas": [1.0], "explicit_delta": 1.0},
            "eval": {"data_json": None, "alpaca": False},
            "lment": {
                "model_key": "full-model-name", "model_device": "cpu",
                "dtype": "fp32", "runs_root": str((root / "runs").resolve()),
                "data": {
                    "concept_json": str(concept.resolve()),
                    "neutral_json": str(neutral.resolve()),
                },
                "features": {
                    "cache_root": str((root / "cache").resolve()),
                    "reuse": False, "fitting_device": "cpu",
                    "max_iterations": 5, "g_sparsity": 0.5, "k_proj": 2,
                },
                "judge": {"model": None},
                "save": {"full_model": False},
                "execution": {"activate_script": None},
            },
        }
        path = root / "source.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_prepare_run_snapshots_inputs_and_writes_absolute_effective_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = prepare_run(self._fixture(root), "Target", execution="windows")
            effective = yaml.safe_load(prepared.effective_config.read_text())
            concept_copy = json.loads(
                (prepared.run_dir / "inputs" / "concept_sentences.json").read_text())

            self.assertRegex(
                prepared.run_dir.name,
                r"^Target_full-model-name_\d{8}_\d{6}$",
            )
            self.assertEqual([item["concept"] for item in concept_copy], ["Target"])
            self.assertTrue(Path(effective["model_name"]).is_absolute())
            self.assertTrue(Path(effective["lment"]["output_dir"]).is_absolute())
            self.assertTrue((prepared.run_dir / "source_config.yaml").is_file())
            self.assertTrue((prepared.run_dir / "run_wrapper.sh").is_file())
            self.assertTrue((prepared.run_dir / "run_wrapper.ps1").is_file())
            shell = (prepared.run_dir / "run_wrapper.sh").read_text()
            powershell = (prepared.run_dir / "run_wrapper.ps1").read_text()
            self.assertIn(str(prepared.effective_config.resolve()), shell)
            self.assertIn("ember.lment_worker", shell)
            self.assertIn(str(prepared.effective_config.resolve()), powershell)
            self.assertIn("ember.lment_worker", powershell)

    def test_run_name_collision_gets_numeric_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixed = datetime(2026, 8, 27, 12, 30, 45)
            with patch("ember.lment_runs.datetime") as mocked_datetime:
                mocked_datetime.now.return_value = fixed
                first = allocate_run_dir(root, "Target", "full-model-name")
                second = allocate_run_dir(root, "Target", "full-model-name")

            self.assertEqual(
                first.name, "Target_full-model-name_20260827_123045")
            self.assertEqual(
                second.name, "Target_full-model-name_20260827_123045_2")

    def test_publish_and_reuse_require_three_exact_matching_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = prepare_run(self._fixture(root), "Target")
            config = prepared.runtime_config
            for directory in feature_concept_dirs(
                    config.features_root, config, "Target").values():
                (directory / "embedding").mkdir(parents=True)
                (directory / "embedding" / "artifact.txt").write_text("value")

            status = publish_run_features(
                config.features_root, config.feature_cache_root, config, "Target")
            self.assertEqual(status, "published")
            cached = validate_feature_bundle(
                config.feature_cache_root, config, "Target")
            self.assertEqual(set(cached), set(FEATURE_BRANCHES))

            neutral = json.loads(Path(config.neutral_json).read_text())
            neutral.append({"sentence": "changed"})
            Path(config.neutral_json).write_text(json.dumps(neutral), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "input mismatch"):
                validate_feature_bundle(config.feature_cache_root, config, "Target")

    def test_partial_shared_cache_fails_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = prepare_run(self._fixture(root), "Target")
            config = prepared.runtime_config
            directories = feature_concept_dirs(
                config.feature_cache_root, config, "Target")
            directories["csvs"].mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "Incomplete shared feature cache"):
                validate_feature_bundle(config.feature_cache_root, config, "Target")
            self.assertFalse(directories["pickles"].exists())

    def test_reuse_copies_validated_feature_bundle_into_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._fixture(root)
            initial = prepare_run(source, "Target")
            config = initial.runtime_config
            for branch, directory in feature_concept_dirs(
                    config.features_root, config, "Target").items():
                directory.mkdir(parents=True)
                (directory / f"{branch}.artifact").write_text(
                    branch, encoding="utf-8")
            self.assertEqual(
                publish_run_features(
                    config.features_root, config.feature_cache_root,
                    config, "Target"),
                "published",
            )

            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            payload["lment"]["features"]["reuse"] = True
            source.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            reused = prepare_run(source, "Target")
            copied = feature_concept_dirs(
                reused.runtime_config.features_root,
                reused.runtime_config,
                "Target",
            )

            for branch, directory in copied.items():
                self.assertEqual(
                    (directory / f"{branch}.artifact").read_text(encoding="utf-8"),
                    branch,
                )
                self.assertTrue((directory / "feature_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
