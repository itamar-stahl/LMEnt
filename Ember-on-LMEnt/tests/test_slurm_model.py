import tempfile
import unittest
from pathlib import Path

import yaml

from ember.slurm_model import (
    SLURM_LMENT_MODEL_PATH,
    SLURM_LMENT_MODEL_PATHS,
    validate_slurm_model_config,
)


class SlurmModelPathTests(unittest.TestCase):
    def _config(self, root: Path, model_name: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / "config.yaml"
        path.write_text(
            yaml.safe_dump({"model_name": model_name}), encoding="utf-8")
        return path

    def test_accepts_every_shared_absolute_control_checkpoint(self) -> None:
        """Each twin pair's control is admissible, and resolves to itself.

        The returned path must be the one that was configured, not the first
        entry -- a Rome run that silently got handed the Pornography pair's
        control would compare two models from different hyperparameters.
        """
        self.assertGreater(len(SLURM_LMENT_MODEL_PATHS), 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, allowed in enumerate(SLURM_LMENT_MODEL_PATHS):
                config = self._config(root / str(index), allowed + "/")
                with self.subTest(model_name=allowed):
                    model = validate_slurm_model_config(
                        config, require_exists=False)
                    self.assertEqual(model.as_posix(), allowed)

    def test_default_is_one_of_the_allowed_checkpoints(self) -> None:
        self.assertIn(SLURM_LMENT_MODEL_PATH, SLURM_LMENT_MODEL_PATHS)

    def test_rejects_relative_symlink_and_ablated_twins(self) -> None:
        """The reasons the guard exists, one case each.

        The symlink is a Windows development path that resolves to nothing on
        the cluster. The other two are ABLATED twins: they were trained with the
        concept held out, so running an erasure method against one measures
        nothing. Widening the allow-list to a second *control* must not let
        either kind through.
        """
        invalid = (
            "../../models_symlinks/win-lment-1b-control-2e",
            "/home/dcor/galbarak2/hf-models/lment-1b-noporn-2e/",
            "/home/dcor/galbarak2/hf-models/lment-1b-norome-2e-b131k/",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, model_name in enumerate(invalid):
                config = self._config(root / str(index), model_name)
                with self.subTest(model_name=model_name):
                    with self.assertRaisesRegex(ValueError, "must use a shared"):
                        validate_slurm_model_config(
                            config, require_exists=False)


if __name__ == "__main__":
    unittest.main()
