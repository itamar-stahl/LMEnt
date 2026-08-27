# LMEnt EMBER feature cache

This directory stores generated SNMF features. The artifacts are large and
ignored by Git; this README is tracked.

For each model, rank, seed, and concept, the main files are:

```text
mf_outputs/
└── <model_key>/
    ├── pickles/rank<R>/seed<S>/<concept>/embedding/
    │   └── embedding.pkl
    ├── csvs/rank<R>/seed<S>/<concept>/embedding/
    │   ├── stats_embed.csv
    │   └── token_features.csv
    └── interpretations/rank<R>/seed<S>/<concept>/embedding/
        ├── potential_features.csv
        └── judge_trace.json          # judge mode only
```

`embedding.pkl`, `stats_embed.csv`, and `token_features.csv` are the reusable
factor cache. If all three exist, `ensure_factor_artifact()` skips SNMF. A new
concept, model key, rank, or seed creates a different cache path.

Prepare one concept on CPU:

```sh
python -m ember.prepare_lment_features \
  --config configs/ember_lment.yaml \
  --concept "Culture of Greece" \
  --concept-json data/concept_sentences.json \
  --neutral-json data/neutral_sentences.json
```

Then run erasure with `--reuse-features`. Threshold or judge selection creates
`potential_features.csv`; judge mode also creates `judge_trace.json`.
