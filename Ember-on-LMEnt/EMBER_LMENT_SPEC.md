# EMBER on LMEnt: practical plan

## Goal

The user provides:

- one LMEnt base model;
- one concept name;
- one concept-sentences JSON;
- one neutral-sentences JSON;
- either an evaluation JSON or an explicit delta;
- a new output directory.

The pipeline returns an erased embedding artifact and `report.json`. It never
writes to the base model.

## Flow

1. Load the named concept and neutral sentences.
2. Load the LMEnt model with Hugging Face.
3. Train the original embedding factorization on CPU.
4. Select concept features:
   - default: two provider-neutral judge callbacks;
   - optional: `--skip-llm-judge --feature-ratio-threshold P`.
5. Select the best delta on `QA_train` and `SimdomQA_train`, or use `--delta`.
6. Apply EMBER only to eligible input-embedding rows.
7. Evaluate `QA_test` and `SimdomQA_test` once.
8. Save, reload, and check the erased result.
9. Optionally run Alpaca prompts as raw LMEnt continuations on CUDA.

## Judge contract

The first callback receives the full feature-description prompt and returns a
description string. The second callback receives the full classification prompt,
including that description, and returns this JSON as a string:

```json
{"is_member": true, "confidence": 0.95}
```

The default callbacks raise `NotImplementedError`. Invalid responses and judge
errors stop the run. Threshold mode does not call either callback. It keeps every
feature with `ratio_abs >= P` and stops if none pass.

## Saving

Default output:

```text
OUTPUT_DIR/
  erased_embeddings.safetensors
  report.json
```

`load_lment_with_erased_embeddings()` loads the base model and replaces its input
embedding in memory after checking the base config, base embedding hash, edited
embedding hash, tensor name, and shape.

`--full-save` keeps the older full-checkpoint option under `OUTPUT_DIR/model`.

## GPU profiles

Alpaca evaluation requires `--gpu-type`:

- `rtx5070-laptop`: FP32, maximum batch size 1;
- `h100`: BF16, maximum batch size 32.

The runtime lowers the batch size when free VRAM is limited. LMEnt receives raw
prompts and returns only new tokens. The user is responsible for suitable prompts.
The scores measure continuation relevance and fluency, not instruction following.

## Tests

- JSON loading and one-concept CLI checks;
- CPU factorization and strict feature selection;
- both judge prompts and strict response parsing;
- pristine restore before every delta and train-only delta selection;
- embedding-only save/load, full save, and source-path protection;
- RTX 5070 and H100 profile checks;
- real RTX 5070 erased-model generation with fake judge callbacks;
- real LMEnt delta search and held-out evaluation;
- complete unit suite in both supported environments.

Run `example.sh` for the normal threshold-mode flow.
