# Standalone EMBER on LMEnt: Implementation and Test Plan

## Goal and scope

Extend the released EMBER code so it can erase any supplied concept from an
LMEnt Hugging Face checkpoint and save a reloadable erased model.

- Use EMBER alone; do not combine it with another erasure method.
- Keep the original sparse factorization, interpretation, feature selection,
  delta grid, and embedding-edit equation.
- A core run is concept-agnostic:
  model + concept + text samples + config → erased checkpoint + report.
- One concept is edited per core run. A concept list repeats the same run from
  pristine weights and produces one independent checkpoint per concept.
- Pornography is only the first full end-to-end acceptance case.

## Verified acceptance model

The complete checkpoint is available at
models_symlinks/win-lment-1b-control-2e. Direct inspection found:

| Property | Actual value | Consequence |
|---|---:|---|
| Architecture | Olmo2ForCausalLM, 18 layers | Use standard Hugging Face interfaces; no TransformerLens adapter |
| Hidden size | 2048 | EMBER factor directions must have dimension 2048 |
| Input embeddings | [100352, 2048], FP32 | Preserve FP32 for the first experiment |
| Output projection | [100352, 2048], FP32, separate tensor | Edit input embeddings only |
| Tokenizer size | 100278 | IDs 100278–100351 are unreachable and must never be edited |
| PAD / BOS / EOS | 100277 / 100257 / 100257 | Do not replace PAD with EOS |
| Context | model 2048, tokenizer metadata 8192 | Cap model inputs at 2048 |
| Cache setting | use_cache=true | Runtime changes must not leak into the saved config |

The tokenizer also contains added control tokens. Eligibility filtering must
exclude every token declared special in tokenizer metadata, not only
BOS/EOS/PAD.

## Pipeline design

    sample source
        → concept texts + neutral texts
        → LMEnt tokenizer integer roles
        → original EMBER embedding factorization
        → original EMBER interpretation
        → original EMBER factored edit
        → base-model evaluation or explicit delta
        → saved/reloaded erased checkpoint + report

### Sampling seam

Use the existing `ConceptDataset` interface.

- `wikipedia_json` is the current implementation: feature training accepts a
  concept JSON path and a neutral JSON path, then passes both to
  `ConceptDataset`.
- Default to the released EMBER Wikipedia data for reproduction.
- Reserve `lment_chunks` for later. It must expose the same dataset behavior,
  so factorization and erasure remain unchanged.
- Do not hardcode Pornography or any of the 18 released concepts.

### Token alignment and feature artifact

- Tokenize concept and neutral texts with the tokenizer loaded from the target
  model.
- Assign integer roles: concept-only, neutral-only, or shared.
- Define V-prime as unique observed IDs that are tokenizer-valid and non-special.
- Factorize the original matrix E[V-prime] transposed with the existing
  SparseMatrixFactorization.
- Add the integer role map and tokenizer size to the existing embedding pickle.
  Keep the current token CSV for interpretation and legacy artifacts.
- During erasure, edit only concept-only IDs with nonzero activation in selected
  features. Never reconstruct eligibility from token strings for new artifacts.

### Evaluation and export

- Reuse option-text continuation likelihood from ember_eval; do not use
  generated letters, Alpaca, or instruction-model evaluation.
- Use per-character-normalized accuracy as primary and retain
  gold-versus-best-distractor margins.
- If evaluation data is supplied, select the delta on QA-train and
  SimdomQA-train, then evaluate QA-test and SimdomQA-test once.
- If a new concept has no evaluation set, require an explicit delta and still
  perform edit/export integrity checks.
- Restore pristine embeddings before every delta. After selection, restore,
  apply the chosen edit once, save with save_pretrained, reload, and report.

## Required code changes

| Area | Minimal change |
|---|---|
| Dataset loading | Expose configurable concept/neutral JSON paths and pass them to the existing ConceptDataset |
| Feature extraction | Make --skip-mlp a true Hugging Face embedding-only path; lazy-load TransformerLens code only for MLP |
| Utilities | Support get_input_embeddings(), Hugging Face token display, complete special-ID filtering, and a separate artifact model_key |
| Feature artifacts | Persist integer token roles and validate model dimension/tokenizer bounds; retain legacy fallback |
| Embedding edit | Consume integer roles and the already-loaded tokenizer; keep the subtraction math unchanged |
| Model loading | Support source/FP32 dtype, preserve PAD and cache configuration, and keep untied output weights untouched |
| LMEnt runner | Add a contained base-model delta/evaluation/export flow that reuses existing EMBER snapshot and apply functions |
| Configuration | Add a general LMEnt config using the repo-relative model symlink, local features, rank/seed/deltas, corpus source, evaluation data, and output root |

SparseMatrixFactorization.fit, embedding ratio statistics,
interpret_features, and the core factored subtraction should not be rewritten.

## Implementation order

1. Add tests and wire the two Wikipedia JSON paths into feature training.
2. Implement the Hugging Face embedding-only feature path and ID-native artifact.
3. Adapt model loading and ID-safe embedding editing.
4. Add the base-model evaluator, delta flow, export, and run report.
5. Run a short real-model smoke test, then the Pornography full acceptance run.

Each step should land only after its fast tests pass.

## Test plan

### Fast tests with a tiny local Olmo2 model

1. Arbitrary concept and neutral files pass through `wikipedia_json`; the
   fixtures do not use Pornography and preserve input order and duplicates.
2. --skip-mlp succeeds when TransformerLens and ActivationGenerator are
   unavailable and emits correctly shaped original-SNMF factors.
3. The feature artifact contains exact integer roles; shared, neutral, special,
   out-of-tokenizer, and zero-activation rows are ineligible.
4. Delta zero is a no-op. A nonzero delta matches the original EMBER equation
   and changes only reported input-embedding rows.
5. lm_head and every non-input-embedding tensor remain unchanged.
6. Model loading preserves FP32, PAD/BOS/EOS, context, cache configuration, and
   untied weights.
7. A golden causal-LM fixture verifies continuation-only log-likelihood and
   per-character normalization.
8. Every grid delta starts from the same pristine checksum; train data alone
   selects the delta; test data runs once afterward.
9. The exported checkpoint reloads through Hugging Face and matches the
   in-memory edited logits and configuration.
10. Existing Gemma/Llama feature artifacts still load through the legacy path.

### Real-model acceptance through models_symlinks

1. Preflight the actual config, tokenizer, embedding/output tensor shapes,
   dtype, special IDs, vocabulary mismatch, and untied weights.
2. Run a short low-rank extraction for a non-Pornography concept to prove the
   full checkpoint path and implementation are concept-agnostic.
3. First full flow: Pornography with the released 300 concept and 300 neutral
   Wikipedia sentences, rank 100, seed 42, ratio threshold 2.0, and deltas
   [0.5, 1, 2, 5, 10, 50, 100, 200].
4. Validate the observed tokenizer partition for that fixture: 5111 IDs,
   comprising 1960 concept-only, 2397 neutral-only, and 754 shared.
5. Interpret features, select on the two 50-item train splits, export/reload,
   and evaluate the two 50-item held-out splits once.
6. Compare all tensors: only reported eligible input rows may differ;
   lm_head, transformer tensors, other input rows, and rows 100278–100351
   must be exact.
7. Report efficacy, specificity, accuracy, and paired margins. Do not require a
   positive erasure effect for the software test to pass.

## Done when

- Any concept in a supplied Wikipedia JSON corpus can produce features and an
  independently erased checkpoint without concept-specific code.
- The full model is loaded from the repo-relative symlink without network access.
- The saved model reloads and passes the tensor-isolation audit.
- Pornography completes the first full train-select-export-test flow.
- Original EMBER behavior for its supported models remains available.

## Deferred

- LMEnt chunk sampling implementation.
- Joint multi-concept edits in one checkpoint.
- OLMo-core changes, MLP erasure, other erasure methods, and relearning.
- Claims about scientific success beyond the reported measurements.
