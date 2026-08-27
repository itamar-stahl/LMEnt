import tempfile
import unittest
from pathlib import Path

import yaml

from ember.slurm_model import (
    SLURM_LMENT_MODEL_PATH,
    validate_slurm_model_config,
)


class SlurmModelPathTests(unittest.TestCase):
    def _config(self, root: Path, model_name: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / "config.yaml"
        path.write_text(
            yaml.safe_dump({"model_name": model_name}), encoding="utf-8")
        return path

    def test_accepts_only_the_shared_absolute_control_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp), SLURM_LMENT_MODEL_PATH + "/")
            model = validate_slurm_model_config(config, require_exists=False)
        self.assertEqual(model.as_posix(), SLURM_LMENT_MODEL_PATH)

    def test_rejects_relative_symlink_and_other_cluster_model(self) -> None:
        invalid = (
            "../../models_symlinks/win-lment-1b-control-2e",
            "/home/dcor/galbarak2/hf-models/lment-1b-noporn-2e/",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, model_name in enumerate(invalid):
                config = self._config(root / str(index), model_name)
                with self.subTest(model_name=model_name):
                    with self.assertRaisesRegex(ValueError, "must use the shared"):
                        validate_slurm_model_config(
                            config, require_exists=False)


if __name__ == "__main__":
    unittest.main()
