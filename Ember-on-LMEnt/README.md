# EMBER

**EMBedding ERasure (EMBER)** is a plug-and-play module for erasing concepts from
language models. It uses Sparse Matrix Factorization to remove concept-related
features directly from the token embeddings, and can be combined with other
unlearning methods (e.g., PISCES, CRISP, RMU, SNMF). Augmenting those methods with
EMBER improves erasure efficacy and specificity and substantially increases
robustness to relearning, with minimal coherence loss.

Paper: [Don't Forget Your Embeddings: Robust Knowledge Erasure via Precise Editing of Embeddings](https://arxiv.org/abs/2606.03695)

This repository covers the full pipeline: extracting and interpreting concept
features, running erasure with any method (with or without EMBER), and evaluating
efficacy, specificity, coherency and robustness to relearning. The released paper
pipeline supports `google/gemma-2-2b-it` and
`meta-llama/Llama-3.1-8B-Instruct`; the standalone runner also supports the local
LMEnt control checkpoint.

[`demo.ipynb`](demo.ipynb) walks through erasing *Harry Potter* from Gemma with
EMBER + SNMF.

## Setup

```bash
git clone https://github.com/itamar-stahl/LMEnt.git
cd LMEnt/Ember-on-LMEnt

conda env create -f environment.yml
conda activate ember
python -m pip install -r requirements.txt

cp .env.example .env   # then add your HF_TOKEN and GEMINI_API_KEY
```

The same `environment.yml` supports Miniforge on Windows and Linux. In
PowerShell, use `Copy-Item .env.example .env` instead of `cp` if `cp` is not
available. Run the setup from **Miniforge Prompt** if PowerShell's execution
policy blocks `Conda.psm1`; no machine-wide policy change is required. The CUDA
12.8 PyTorch wheel supports RTX 50-series GPUs; CPU-only machines can still run
the unit tests, with CUDA integration tests skipped.

For the original Gemma/Llama paper methods, first complete the core setup above,
then additionally install `python -m pip install -e ".[paper]"`.

Verify the installed core environment with:

```bash
python -c "import site, sys, torch, ember, factorization; from pathlib import Path; assert not site.ENABLE_USER_SITE; assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); print(sys.executable, torch.__version__)"
python -m pip check
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q ember tests
```

To include the real-checkpoint CUDA test, set `EMBER_LMENT_MODEL_PATH` first.
On Linux:

```bash
export EMBER_LMENT_MODEL_PATH=/path/to/lment-1b-control-2e
```

On Windows PowerShell:

```powershell
$env:EMBER_LMENT_MODEL_PATH = "C:\path\to\lment-1b-control-2e"
```

The vendored method sources live in `external/` (snmf, CRISP, wmdp, PISCES).
`HF_TOKEN` is needed to download gated models (e.g. Llama-3.1-8B-Instruct);
`GEMINI_API_KEY` is needed for feature interpretation and the Alpaca coherence judge
(the erasure grids run without it). See `.env.example` for details.

## Concept features

Erasure reads precomputed concept features from `mf_outputs/`. Each concept has two
factorizations:

- **Embedding features**: a sparse factorization of the token-embedding matrix. This
  is all EMBER needs.
- **MLP features**: Semi-NMF over MLP activations, used by the SNMF erasure method.

Each is paired with an LLM-written interpretation that selects the concept-related
features (`potential_features.csv`).

### Provided features (recommended)

We publish all features for both models on the Hugging Face Hub:
[**ClSu/ember-features**](https://huggingface.co/datasets/ClSu/ember-features).
Erasure runs download them automatically (scoped to the run's model, rank, seed and
concepts; files already present are skipped), so no manual step is needed. To
pre-fetch:

```python
from huggingface_hub import snapshot_download
snapshot_download("ClSu/ember-features", repo_type="dataset", local_dir="mf_outputs",
                  allow_patterns=["google_gemma-2-2b-it/**/Harry_Potter/**"])
```

### Re-training the features

You can also regenerate the features. The factorization step is seeded and
reproducible; the interpretation step calls an LLM and may select a slightly
different feature set between runs.

The training has two tracks. EMBER only needs the **embedding** track; add the
**MLP** track as well if you also want to run SNMF erasure.
Matrix factorization runs on CPU by default, matching the original factorization
path. The model itself may still be loaded on CUDA independently.

```bash
# Factorize. Drop --skip-mlp to also build the MLP track (for SNMF).
python -m ember.train_mf_features --concepts "Harry Potter" --ranks 100 \
    --model-name google/gemma-2-2b-it --seed 42 --skip-mlp

# Interpret and select concept features (needs GEMINI_API_KEY).
python -m ember.interpret_features --concepts "Harry Potter" --rank 100 \
    --model-name google/gemma-2-2b-it --seed 42 --tracks embedding
```

To use another Wikipedia sentence corpus, pass its concept and neutral JSON files:

```bash
python -m ember.train_mf_features --concepts "Any Concept" --ranks 100 \
    --model-name /path/to/model --skip-mlp \
    --concept-json /path/to/concept_sentences.json \
    --neutral-json /path/to/neutral_sentences.json
```

The concept file is a list of `{ "concept": ..., "sentences": [...] }` records.
The neutral file is a list of records with a `"sentence"` field. The existing
files in `data/` remain the defaults.

## Standalone EMBER on LMEnt

The LMEnt runner applies only the EMBER embedding edit to one named concept in a
local control model. The concept and neutral JSON paths are always explicit:

```bash
python -m ember.run_lment_ember --config configs/ember_lment.yaml \
    --concept "Culture of Greece" \
    --concept-json data/concept_sentences.json \
    --neutral-json data/neutral_sentences.json \
    --eval-json data/mc_questions.json \
    --output-dir lment_outputs/culture-of-greece \
    --skip-llm-judge --feature-ratio-threshold 8.0
```

`--skip-llm-judge` requires `--feature-ratio-threshold P`. It selects every
feature where `ratio_abs = mean(|G| concept) / mean(|G| neutral)` is at least P.
The run stops if no feature passes. Judge mode is the default and uses two
provider-neutral Python callbacks in `ember/judge_callbacks.py`:

1. `describe_feature(full_prompt) -> description_string`
2. `classify_feature(full_prompt_with_description) -> JSON_string`

The second JSON string must contain `{"is_member": bool, "confidence": 0..1}`.
The default callbacks raise an informative `NotImplementedError`; implement them
or inject callbacks through `run_lment_pipeline()`. Judge errors stop the run.
For example:

```python
from ember.lment_pipeline import run_lment_pipeline

report = run_lment_pipeline(
    config,
    concept="Culture of Greece",
    describe_callback=lambda full_prompt: provider.describe(full_prompt),
    classify_callback=lambda full_prompt: provider.classify(full_prompt),
)
```

Automatic best-delta selection requires an evaluation JSON containing
`QA_train`, `SimdomQA_train`, `QA_test`, and `SimdomQA_test` for the concept.
The train splits select delta; the test splits run once afterward. For a concept
without these questions, pass an explicit `--delta`.

By default the output directory contains only `erased_embeddings.safetensors`
and `report.json`. The source checkpoint is never written. Load it with:

```python
from ember.erased_embedding import load_lment_with_erased_embeddings

model, tokenizer = load_lment_with_erased_embeddings(
    "/path/to/base-lment",
    "/path/to/output/erased_embeddings.safetensors",
    device="cuda",
)
```

Add `--full-save` to write a complete Hugging Face checkpoint under
`OUTPUT_DIR/model` instead.

Alpaca evaluation supports `--gpu-type rtx5070-laptop` (maximum batch 1) and
`--gpu-type h100` (maximum batch 32), then lowers the batch using currently free
VRAM. LMEnt receives each prompt as raw text, and only newly generated tokens are
returned. These results are prompt-continuation relevance/fluency, not an
instruction-following claim. Alpaca relevance and fluency also use the documented
provider-neutral callbacks in `ember/judge_callbacks.py`.
Each Alpaca callback receives the complete scoring prompt and must return text
containing `Rating: [[0]]`, `Rating: [[1]]`, or `Rating: [[2]]`.

See `example.sh` for a complete threshold-mode run.

For a real-checkpoint mechanical smoke test without concept evaluation, run
`tests/lment_erasure_smoke.py`. Its `--keep-erased-model` flag preserves the
otherwise-temporary erased checkpoint and prints a JSON report with its path.

For Llama, use rank 200 and its model name:

```bash
python -m ember.train_mf_features --concepts "Harry Potter" --ranks 200 \
    --model-name meta-llama/Llama-3.1-8B-Instruct --seed 42 --skip-mlp
python -m ember.interpret_features --concepts "Harry Potter" --rank 200 \
    --model-name meta-llama/Llama-3.1-8B-Instruct --seed 42 --tracks embedding
```

To use whatever is already in `mf_outputs/` and never contact the Hub, pass
`--features-source local` to the erasure command (below) or set
`features_source: local` in the config.

## Running erasure

Each run is driven by a YAML config in `configs/` plus CLI overrides:

```bash
python -m ember.run_erasure --config configs/snmf_ember_gemma.yaml \
    --concepts "Harry Potter" --train-eval mc
```

**Methods** (the `method` field / config prefix): `snmf`, `rmu`, `crisp`, `pisces`,
`ember`. Config names follow `<method>_<model>.yaml` for the method alone and
`<method>_ember_<model>.yaml` to augment it with the EMBER embedding edit, where
`<model>` is `gemma` or `llama`. For example, `configs/rmu_ember_llama.yaml` runs RMU
with EMBER on Llama.

**Modes**: `--train-eval {mc,open}` selects the question format that drives the grid
search. The pipeline runs a grid search, validates the top configurations, evaluates
the best one on the held-out test set, and (if enabled) measures relearning.

**Results** are written under:

```
results/<method>[_ef]/<model>/rank<R>/seed<S>/
    train_<mode>/   # grid search + validation
    test_<mode>/    # final-test metrics and relearning
```

The `_ef` suffix marks runs that used EMBER as a pre-step (method + EMBER). Metrics
include efficacy, specificity, MMLU retention, Alpaca instruction-following and
fluency, the harmonic aggregate, and post-relearning accuracy.

## Citation

```bibtex
@misc{suslik2026dontforgetembeddingsrobust,
  title         = {Don't Forget Your Embeddings: Robust Knowledge Erasure via Precise Editing of Embeddings},
  author        = {Clara Haya Suslik and Or Shafran and Mor Geva},
  year          = {2026},
  eprint        = {2606.03695},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.03695},
}
```
