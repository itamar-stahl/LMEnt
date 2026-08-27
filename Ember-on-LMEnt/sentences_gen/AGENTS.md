# sentences_gen — agent guide

Turns an Untaught blacklist (Wikidata QIDs) into an EMBER `concept_sentences.json`
by harvesting one on-topic sentence per corpus chunk. The module docstring in
`blacklist_to_concept_sentences.py` carries the algorithm; `--help` carries the
flags. This file carries what neither can tell you.

## Run it on the login node

Elasticsearch runs as one process on **c-003** and is reachable from there only.
Everything here needs it: chunk resolution, chunk text, both live test suites.
A GPU node cannot do any of it.

```sh
. Untaught/activate_env.sh                          # conda, PYTHONPATH, ES_PASSWORD
cd "$LMENT_ROOT/Ember-on-LMEnt/sentences_gen"       # activate_env.sh cd's to Untaught
python tests/run_tests.py                           # 3 suites; run before trusting a corpus
python blacklist_to_concept_sentences.py ../../Untaught/blacklists/<subject>.json
```

Defaults are the recommendation. Reach for a flag when a run fails and the
rejection histogram says which one.

**Done** when the run folder holds `concept_sentences.json`. Its absence means
the distribution gate rejected the corpus — the run failed, and the diagnostics
are in the same folder.

## Each run is self-contained

`outputs/<blacklist-stem>_<date>_<time>/` gets the copied blacklist, the resolved
chunk ids with every parameter, the per-sentence provenance, and the corpus. It
is gitignored, and nothing is written anywhere else.

**`data/concept_sentences.json` stays untouched by design.** Point the consumer at
the run folder instead:

```sh
python -m ember.train_mf_features --concept-json outputs/<run>/concept_sentences.json
```

## When it fails

Read `chunk_ids.json` → `harvest.rejections` first. It names the gate that ate the
pool.

| Reason | Meaning | Action |
|---|---|---|
| `no_qualifying_mention` | chunk's mentions all fall under the thresholds | expected in volume; a large share plus a short corpus means the pool is genuinely thin |
| `offset_mismatch` | entity offsets do not index into the stored text | **stop and investigate** — see Invariants |
| `truncated_span` | mention sits in a chunk-edge fragment | expected, low single digits |
| `duplicate` | sentence already harvested | expected; Wikipedia repeats itself |
| `length_out_of_range` | outside 5–120 words | expected, low |

A distribution-gate failure prints which of median / stdev / distinct-count
missed, against the shipped 18-concept reference. Treat it as a real signal about
the corpus rather than a threshold to widen.

If a subject cannot reach 300, the honest fixes are a better QID set (see
`Untaught/CHOOSING_A_SUBJECT.md`) or fewer sentences via `-n`.

## Invariants

1. **`offset_mismatch` is a deployment alarm, not noise.** Entity offsets are
   rebased chunk-relative by `OLMo-core numpy_dataset._get_entities_within_range`
   against the chunk's slice of the source document, while the index stores
   `decode(input_ids)`. They agree only if the tokenizer round-trip is lossless.
   A nonzero count means it is not, on this deployment. The module rejects those
   chunks rather than harvesting a plausible wrong sentence, so the cost is yield,
   not corruption. Run `tests/test_live_es.py` for the measured alignment rate.
2. **One `DEFAULT_THRESHOLDS` gates both retrieval and selection.** A source can
   never fetch a chunk it is not also allowed to anchor a sentence with. Keep any
   new filter reading that same resolved dict.
3. **The thresholds sit above the paper's on purpose** (`entity_linking` 0.7 vs
   0.6, coref 0.95 vs 0.6). Untaught retrieves broadly because it drops chunks
   from training; we need a sentence carrying the concept's own tokens. Restoring
   the paper's values silently widens what lands in the corpus.
4. **Same seed, same corpus.** Sampling is a per-QID seeded shuffle drawn
   round-robin. Changing the seed, the thresholds, or the QID set changes the
   output; nothing else should.
5. **The derived concept name is title-cased with a small-word stoplist**, so
   `harry_potter` → `Harry Potter` and `republic_of_ireland` → `Republic of
   Ireland`. It has no special case for `covid-19_pandemic` → `COVID-19 pandemic`.
   Check the derived name against your QA JSON keys before relying on it
   downstream: a mismatch surfaces much later as an empty evaluation set.

## Tests

`tests/run_tests.py` runs three suites: `units` (offline, runs anywhere),
`live_es` (deployment invariants), `end_to_end` (the CLI against the real index).
`SENTENCES_GEN_E2E_FULL=0` skips the 300-sentence run while iterating.

Without Elasticsearch the live suites **skip** and the run still passes — skips
are preconditions, and each states which one is missing. Keep `units` free of
Elasticsearch so it stays runnable off-cluster.
