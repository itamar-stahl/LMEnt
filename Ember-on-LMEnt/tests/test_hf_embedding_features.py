import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNMF_ROOT = PROJECT_ROOT / "external" / "snmf"
if str(SNMF_ROOT) not in sys.path:
    sys.path.insert(0, str(SNMF_ROOT))


def _write_tiny_olmo2_checkpoint(path: Path) -> None:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "shared": 4,
        "concept": 5,
        "neutral": 6,
        "extra": 7,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)

    config = Olmo2Config(
        vocab_size=10,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        use_cache=True,
    )
    Olmo2ForCausalLM(config).save_pretrained(path)


class HuggingFaceEmbeddingFeatureTests(unittest.TestCase):
    def test_interpretation_cli_exposes_model_key_without_google_dependency(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            str(PROJECT_ROOT),
            str(PROJECT_ROOT / "external" / "snmf"),
        ])
        result = subprocess.run(
            [sys.executable, "-m", "ember.interpret_features", "--help"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("--model-key", result.stdout)

    def test_skip_mlp_runs_without_transformer_lens_and_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            output_dir = root / "features"
            model_dir.mkdir()
            _write_tiny_olmo2_checkpoint(model_dir)

            concept_json = root / "concept.json"
            neutral_json = root / "neutral.json"
            concept_json.write_text(json.dumps([
                {"concept": "Arbitrary concept", "sentences": ["concept shared <bos>"]},
            ]), encoding="utf-8")
            neutral_json.write_text(json.dumps([
                {"sentence": "neutral shared"},
            ]), encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([
                str(PROJECT_ROOT),
                str(PROJECT_ROOT / "external" / "snmf"),
            ])
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ember.train_mf_features",
                    "--concepts",
                    "Arbitrary concept",
                    "--ranks",
                    "2",
                    "--seed",
                    "7",
                    "--model-name",
                    str(model_dir),
                    "--model-key",
                    "tiny-olmo2",
                    "--model-device",
                    "cpu",
                    "--fitting-device",
                    "cpu",
                    "--concept-json",
                    str(concept_json),
                    "--neutral-json",
                    str(neutral_json),
                    "--skip-mlp",
                    "--max-iterations",
                    "2",
                    "--g-sparsity",
                    "1.0",
                    "--k-proj",
                    "2",
                    "--outdir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            artifact = (
                output_dir
                / "tiny-olmo2"
                / "pickles"
                / "rank2"
                / "seed7"
                / "Arbitrary_concept"
                / "embedding"
                / "embedding.pkl"
            )
            self.assertTrue(artifact.is_file())
            with artifact.open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(tuple(payload["nmf"].F_.shape), (8, 2))
            self.assertEqual(payload["artifact_version"], 2)
            self.assertEqual(payload["model_key"], "tiny-olmo2")
            self.assertEqual(payload["tokenizer_size"], 8)
            self.assertEqual(payload["embedding_vocab_size"], 10)
            self.assertEqual(payload["embedding_dim"], 8)
            self.assertEqual(payload["vprime_token_ids"], [4, 5, 6])
            self.assertEqual(payload["token_roles"], {4: 2, 5: 1, 6: 0})


if __name__ == "__main__":
    unittest.main()
