# LMEnt EMBER shared feature cache

Successful runs may publish their embedding-feature bundle here. Large cache
artifacts are ignored by Git; this README is tracked.

```text
mf_outputs/<model-key>/
├── csvs/rank<R>/seed<S>/<concept>/
├── interpretations/rank<R>/seed<S>/<concept>/
└── pickles/rank<R>/seed<S>/<concept>/
```

Each of the three concept folders contains copies of:

```text
concept_sentences.json
neutral_sentences.json
feature_manifest.json
```

The concept file contains only the relevant concept record. The neutral file
contains the complete neutral input. The manifest records the model marker,
rank, seed, and factorization parameters.

With `lment.features.reuse: true`, all three branches must exist and every
provenance file must match exactly. Missing, partial, or changed caches stop the
run with an informative error; the runner never overwrites them. With reuse
disabled, features are built inside the run folder and copied here only if all
three destination branches are absent.
