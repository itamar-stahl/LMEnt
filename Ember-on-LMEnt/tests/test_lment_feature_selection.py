import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ember.lment_feature_selection import select_by_threshold, select_with_judge
from ember import judge_callbacks


class LMEntFeatureSelectionTests(unittest.TestCase):
    def test_default_callbacks_raise_informative_errors(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "description judge"):
            judge_callbacks.describe_feature("full prompt")
        with self.assertRaisesRegex(NotImplementedError, "classification judge"):
            judge_callbacks.classify_feature("full prompt")

    def _write_inputs(self, root: Path) -> None:
        common = Path("rank3") / "seed7" / "Any_concept" / "embedding"
        csv_dir = root / "local-model" / "csvs" / common
        csv_dir.mkdir(parents=True)
        pd.DataFrame([
            {"feature": 0, "ratio_abs": 2.0, "mean_abs_concept": 1.0,
             "mean_abs_neutral": 0.5, "num_concept": 2, "num_neutral": 2},
            {"feature": 1, "ratio_abs": 9.0, "mean_abs_concept": 0.9,
             "mean_abs_neutral": 0.1, "num_concept": 3, "num_neutral": 1},
            {"feature": 2, "ratio_abs": 12.0, "mean_abs_concept": 1.2,
             "mean_abs_neutral": 0.1, "num_concept": 4, "num_neutral": 1},
        ]).to_csv(csv_dir / "stats_embed.csv", index=False)
        pd.DataFrame([
            {"feature": 0, "activating_tokens": "['low']"},
            {"feature": 1, "activating_tokens": "['alpha', 'beta']"},
            {"feature": 2, "activating_tokens": "['gamma']"},
        ]).to_csv(csv_dir / "token_features.csv", index=False)

    def test_threshold_requires_features_at_or_above_p(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_inputs(root)
            result = select_by_threshold(
                features_root=root, model_key="local-model",
                concept="Any concept", rank=3, seed=7, threshold=10.0)
            selected = pd.read_csv(result.potential_features_path)
            self.assertEqual(result.selected_feature_ids, [2])
            self.assertEqual(selected["selection_mode"].tolist(), ["threshold"])
            with self.assertRaisesRegex(ValueError, "No embedding feature"):
                select_by_threshold(
                    features_root=root, model_key="local-model",
                    concept="Any concept", rank=3, seed=7, threshold=20.0)

    def test_judge_receives_full_prompts_and_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_inputs(root)
            description_prompts = []
            classification_prompts = []

            def describe(prompt: str) -> str:
                description_prompts.append(prompt)
                return (
                    "A gamma-specific concept."
                    if "gamma" in prompt else "An alpha-specific concept.")

            def classify(prompt: str) -> str:
                classification_prompts.append(prompt)
                return json.dumps({"is_member": "gamma" in prompt, "confidence": 0.95})

            result = select_with_judge(
                features_root=root, model_key="local-model",
                concept="Any concept", rank=3, seed=7,
                prefilter_threshold=8.0, confidence_threshold=0.9, top_k=2,
                describe_callback=describe, classify_callback=classify)
            self.assertEqual(result.selected_feature_ids, [2])
            self.assertEqual(len(description_prompts), 2)
            self.assertIn("Tokens:", description_prompts[0])
            self.assertIn("Description: An alpha-specific concept.", classification_prompts[0])
            trace = json.loads(Path(result.judge_trace_path).read_text(encoding="utf-8"))
            self.assertEqual(len(trace), 2)

    def test_judge_rejects_non_boolean_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_inputs(root)
            with self.assertRaisesRegex(ValueError, "JSON boolean"):
                select_with_judge(
                    features_root=root, model_key="local-model",
                    concept="Any concept", rank=3, seed=7,
                    prefilter_threshold=8.0, confidence_threshold=0.9, top_k=2,
                    describe_callback=lambda _prompt: "Description",
                    classify_callback=lambda _prompt: json.dumps({
                        "is_member": "false", "confidence": 0.9}),
                )


if __name__ == "__main__":
    unittest.main()
