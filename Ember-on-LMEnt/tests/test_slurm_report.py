import json
import tempfile
import unittest
from pathlib import Path

from slurm.tests.verify_e2e_report import verify_e2e_report


class SlurmReportTests(unittest.TestCase):
    def test_accepts_complete_h100_judge_erasure_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir()
            report = outputs / "report.json"
            out = root / "log.out"
            err = root / "log.err"
            report.write_text(json.dumps({
                "feature_selection": {"mode": "judge", "selected_feature_ids": [3]},
                "save": {"mode": "embedding_only"},
                "integrity": {"passed": True},
                "feature_cache": {"status": "published"},
                "alpaca": {
                    "n": 1,
                    "max_items": 1,
                    "gpu": "NVIDIA H100 80GB HBM3",
                    "mean_relevance": 2.0,
                    "mean_fluency": 1.0,
                },
            }), encoding="utf-8")
            out.write_text(
                '{"conda_env": "lment", "gpu": "NVIDIA H100 80GB HBM3"}',
                encoding="utf-8",
            )
            err.write_text("A harmless warning", encoding="utf-8")
            (root / "run_environment.json").write_text("{}", encoding="utf-8")

            result = verify_e2e_report(report, out, err)

        self.assertEqual(result["selected_features"], [3])
        self.assertEqual(result["alpaca_items"], 1)

    def test_rejects_wrong_gpu_or_failed_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir()
            report = outputs / "report.json"
            out = root / "log.out"
            err = root / "log.err"
            report.write_text(json.dumps({
                "feature_selection": {"mode": "judge", "selected_feature_ids": [1]},
                "save": {"mode": "embedding_only"},
                "integrity": {"passed": False},
                "feature_cache": {"status": "published"},
                "alpaca": {"n": 1, "max_items": 1,
                            "gpu": "NVIDIA H100 80GB HBM3"},
            }), encoding="utf-8")
            out.write_text('{"conda_env": "lment", "gpu": "RTX 5070"}',
                           encoding="utf-8")
            err.write_text("", encoding="utf-8")
            (root / "run_environment.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(AssertionError):
                verify_e2e_report(report, out, err)


if __name__ == "__main__":
    unittest.main()
