# LMEnt EMBER outputs

This directory stores erased LMEnt artifacts and evaluation reports. Generated
outputs can be large and are ignored by Git; this README is tracked.

The default output for one run is:

```text
lment_outputs/<run-name>/
├── erased_embeddings.safetensors
└── report.json
```

- `erased_embeddings.safetensors` contains only the modified input-embedding
  layer and metadata linking it to the original LMEnt checkpoint.
- `report.json` records the concept, selected features, delta, edited token IDs,
  evaluation results, save path, and integrity checks.

Load the result without modifying the base model:

```python
from ember.erased_embedding import load_lment_with_erased_embeddings

model, tokenizer = load_lment_with_erased_embeddings(
    "/path/to/base-lment",
    "lment_outputs/<run-name>/erased_embeddings.safetensors",
    device="cuda",
)
```

With `--full-save`, the run instead also writes a complete Hugging Face model
under `<run-name>/model/`. Each command must use a new `--output-dir`; the
pipeline never overwrites the original model or an existing output directory.
