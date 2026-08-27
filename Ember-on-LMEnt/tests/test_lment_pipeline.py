import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNMF_ROOT = PROJECT_ROOT / "external" / "snmf"
if str(SNMF_ROOT) not in sys.path:
    sys.path.insert(0, str(SNMF_ROOT))

from ember.erasure.model_loader import load_local_causal_lm
from ember.erased_embedding import load_lment_with_erased_embeddings
from ember.lment_pipeline import LMEntRunConfig, run_concept, run_lment_pipeline
from tests.test_hf_embedding_features import _write_tiny_olmo2_checkpoint


def _write_embedding_artifact(root: Path) -> None:
    pickle_dir = (
        root / "tiny-olmo2" / "pickles" / "rank1" / "seed42"
        / "Arbitrary_concept" / "embedding"
    )
    interpretation_dir = (
        root / "tiny-olmo2" / "interpretations" / "rank1" / "seed42"
        / "Arbitrary_concept" / "embedding"
    )
    pickle_dir.mkdir(parents=True)
    interpretation_dir.mkdir(parents=True)
    nmf = SimpleNamespace(
        F_=torch.ones((8, 1), dtype=torch.float32),
        G_=torch.tensor([[2.0]], dtype=torch.float32),
    )
    with (pickle_dir / "embedding.pkl").open("wb") as handle:
        pickle.dump({
            "artifact_version": 2,
            "model_key": "tiny-olmo2",
            "nmf": nmf,
            "vprime_token_ids": [5],
            "token_roles": {5: 1},
            "tokenizer_size": 8,
            "embedding_vocab_size": 10,
            "embedding_dim": 8,
        }, handle)
    pd.DataFrame([{"feature": 0, "metric_score": 3.0}]).to_csv(
        interpretation_dir / "potential_features.csv", index=False)
    csv_dir = (
        root / "tiny-olmo2" / "csvs" / "rank1" / "seed42"
        / "Arbitrary_concept" / "embedding"
    )
    csv_dir.mkdir(parents=True)
    pd.DataFrame([{
        "feature": 0,
        "ratio_abs": 3.0,
        "mean_abs_concept": 1.0,
        "mean_abs_neutral": 1.0 / 3.0,
        "num_concept": 1,
        "num_neutral": 1,
    }]).to_csv(csv_dir / "stats_embed.csv", index=False)
    pd.DataFrame([{
        "feature": 0,
        "activating_tokens": "['concept']",
    }]).to_csv(csv_dir / "token_features.csv", index=False)


class LMEntPipelineTests(unittest.TestCase):
    def test_callback_judge_is_deployable_through_full_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "source-model"
            features_root = root / "features"
            output_dir = root / "judged-output"
            model_path.mkdir()
            _write_tiny_olmo2_checkpoint(model_path)
            _write_embedding_artifact(features_root)
            prompts = []

            report = run_lment_pipeline(LMEntRunConfig(
                model_path=model_path,
                model_key="tiny-olmo2",
                features_root=features_root,
                runs_root=root / "unused",
                output_dir=output_dir,
                rank=1,
                explicit_delta=0.5,
                device="cpu",
                selection_mode="judge",
                ratio_thresh=2.0,
            ), concept="Arbitrary concept",
                describe_callback=lambda prompt: (
                    prompts.append(prompt) or "The arbitrary concept."),
                classify_callback=lambda prompt: (
                    prompts.append(prompt)
                    or '{"is_member": true, "confidence": 0.99}'),
            )

            self.assertEqual(report["feature_selection"]["mode"], "judge")
            self.assertEqual(report["artifact"]["selected_feature_ids"], [0])
            self.assertEqual(len(prompts), 2)
            self.assertTrue(Path(
                report["feature_selection"]["judge_trace_path"]).is_file())

    def test_rejects_export_inside_source_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source-model"
            source.mkdir()
            config = LMEntRunConfig(
                model_path=source,
                model_key="tiny-olmo2",
                features_root=Path(tmp) / "features",
                runs_root=source,
                explicit_delta=0.5,
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                run_concept(config, concept="Arbitrary concept")

    def test_automatic_selection_uses_train_and_pairs_held_out_margins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "source-model"
            features_root = root / "features"
            output_root = root / "outputs"
            eval_path = root / "mc.json"
            model_path.mkdir()
            _write_tiny_olmo2_checkpoint(model_path)
            _write_embedding_artifact(features_root)
            question = {
                "q": "Which option is correct?",
                "correct_answer": "one",
                "options": ["one", "two", "three", "four"],
            }
            eval_path.write_text(json.dumps({
                "Arbitrary concept": {
                    "QA_train": [question],
                    "SimdomQA_train": [question],
                    "QA_test": [question],
                    "SimdomQA_test": [question],
                },
            }), encoding="utf-8")

            calls = []

            def fake_evaluate(_model, _tokenizer, items, *, include_records):
                split = items["qa"][0].split
                occurrence = sum(call[0] == split for call in calls)
                calls.append((split, include_records))
                is_baseline = occurrence == 0
                qa_accuracy = 0.75 if is_baseline else 0.25
                margin = 0.4 if is_baseline else -0.2
                result = {}
                for subset in ("qa", "simdom"):
                    entry = {
                        "n": 1,
                        "accuracy": qa_accuracy if subset == "qa" else 0.75,
                        "mean_margin": margin,
                    }
                    if include_records:
                        entry["records"] = [{
                            "question": "Which option is correct?",
                            "correct_letter": "A",
                            "margin": margin,
                        }]
                    result[subset] = entry
                return result

            with patch(
                    "ember.lment_pipeline._evaluate_pair",
                    side_effect=fake_evaluate):
                report = run_concept(LMEntRunConfig(
                    model_path=model_path,
                    model_key="tiny-olmo2",
                    features_root=features_root,
                    runs_root=output_root,
                    rank=1,
                    seed=42,
                    ratio_thresh=2.0,
                    deltas=(1.0, 0.5),
                    eval_json=eval_path,
                    device="cpu",
                    dtype="fp32",
                ), concept="Arbitrary concept")

        self.assertEqual(report["chosen_delta"], 0.5)
        self.assertEqual([split for split, _ in calls], [
            "train", "train", "train", "test", "test",
        ])
        self.assertEqual([include for split, include in calls if split == "test"], [
            True, True,
        ])
        self.assertAlmostEqual(
            report["evaluation"]["test"]["paired_margins"]["qa"]["mean_delta"],
            -0.6,
        )
        self.assertTrue(report["integrity"]["passed"])

    def test_explicit_delta_without_eval_exports_reloadable_isolated_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "source-model"
            features_root = root / "features"
            output_root = root / "outputs"
            model_path.mkdir()
            _write_tiny_olmo2_checkpoint(model_path)
            _write_embedding_artifact(features_root)

            report = run_concept(LMEntRunConfig(
                model_path=model_path,
                model_key="tiny-olmo2",
                features_root=features_root,
                runs_root=output_root,
                rank=1,
                seed=42,
                ratio_thresh=2.0,
                explicit_delta=0.5,
                eval_json=None,
                device="cpu",
                dtype="fp32",
            ), concept="Arbitrary concept")

            erased_path = (
                output_root / "Arbitrary_concept" / "erased_embeddings.safetensors")
            report_exists = (erased_path.parent / "report.json").is_file()
            source, _ = load_local_causal_lm(model_path, device="cpu")
            erased, _ = load_lment_with_erased_embeddings(
                model_path, erased_path, device="cpu")
            other_model_path = root / "other-source-model"
            other_model_path.mkdir()
            _write_tiny_olmo2_checkpoint(other_model_path)
            with self.assertRaisesRegex(ValueError, "base embedding weights"):
                load_lment_with_erased_embeddings(
                    other_model_path, erased_path, device="cpu")

        source_state = source.state_dict()
        erased_state = erased.state_dict()
        embedding_name = "model.embed_tokens.weight"
        changed_rows = torch.where(
            (source_state[embedding_name] != erased_state[embedding_name]).any(dim=1)
        )[0].tolist()
        self.assertEqual(changed_rows, [5])
        for name, tensor in source_state.items():
            if name != embedding_name:
                self.assertTrue(torch.equal(tensor, erased_state[name]), msg=name)
        self.assertEqual(report["edit"]["edited_token_ids"], [5])
        self.assertEqual(report["chosen_delta"], 0.5)
        self.assertTrue(report["integrity"]["passed"])
        self.assertTrue(report["integrity"]["reload_logits_match"])
        self.assertEqual(report["save"]["mode"], "embedding_only")
        self.assertEqual(Path(report["erased_embeddings_path"]), erased_path)
        self.assertTrue(report_exists)

    def test_full_save_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "source-model"
            features_root = root / "features"
            output_root = root / "outputs"
            model_path.mkdir()
            _write_tiny_olmo2_checkpoint(model_path)
            _write_embedding_artifact(features_root)
            report = run_concept(LMEntRunConfig(
                model_path=model_path,
                model_key="tiny-olmo2",
                features_root=features_root,
                runs_root=output_root,
                rank=1,
                explicit_delta=0.5,
                device="cpu",
                full_save=True,
            ), concept="Arbitrary concept")
            checkpoint = Path(report["checkpoint_path"])
            self.assertTrue((checkpoint / "config.json").is_file())
            self.assertEqual(report["save"]["mode"], "full_model")


if __name__ == "__main__":
    unittest.main()
