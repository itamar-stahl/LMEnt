import json
import tempfile
import unittest
from pathlib import Path

from ember.local_datasets import ConceptDataset, DATA_DIR


class WikipediaJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.concept_json = root / "concepts.json"
        self.neutral_json = root / "neutral.json"

        self.concept_json.write_text(
            json.dumps([
                {"concept": "Concept A", "sentences": ["a1", "a1", "a2"]},
                {"concept": "Concept B", "sentences": ["b1"]},
            ]),
            encoding="utf-8",
        )
        self.neutral_json.write_text(
            json.dumps([
                {"url": "https://example/1", "title": "One", "sentence": "n1"},
                {"url": "https://example/2", "title": "Two", "sentence": "n2"},
            ]),
            encoding="utf-8",
        )

    def test_loads_requested_concept_and_neutral_sentences(self) -> None:
        dataset = ConceptDataset(
            "Concept A",
            concept_path=self.concept_json,
            neutral_path=self.neutral_json,
        )

        self.assertEqual(
            dataset.as_forget_retain(),
            {"forget": ["a1", "a1", "a2"], "retain": ["n1", "n2"]},
        )

    def test_missing_concept_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Concept"):
            ConceptDataset(
                "Unknown Concept",
                concept_path=self.concept_json,
                neutral_path=self.neutral_json,
            )

    def test_non_list_json_is_rejected(self) -> None:
        self.concept_json.write_text(json.dumps({"concept": "Concept A"}), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Expected list"):
            ConceptDataset(
                "Concept A",
                concept_path=self.concept_json,
                neutral_path=self.neutral_json,
            )

    def test_released_non_pornography_concept_has_expected_counts(self) -> None:
        dataset = ConceptDataset(
            "Culture of Greece",
            concept_path=DATA_DIR / "concept_sentences.json",
            neutral_path=DATA_DIR / "neutral_sentences.json",
        )

        samples = dataset.as_forget_retain()
        self.assertEqual(len(samples["forget"]), 300)
        self.assertEqual(len(samples["retain"]), 300)


if __name__ == "__main__":
    unittest.main()
