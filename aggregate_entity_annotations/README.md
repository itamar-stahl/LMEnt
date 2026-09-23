# aggregate_entity_annotations — one annotation per mention, from four sources

Step 3 of *Annotating Pretraining Data* in the [root README](../README.md). It reads
one NDJSON file written by
[`third_party/maverick-coref/run_maverick.py`](../third_party/maverick-coref/run_maverick.py)
and writes the same records back with the four raw annotation fields removed and
a single `entities` list added: one entry per character span, each carrying the
Wikidata candidates for that span and a score per evidence source.

Nothing here runs a model. Every number is arithmetic over annotations ReFinED
and Maverick already produced, plus one LMDB database consulted to turn a QID
into a name.

    __init__.py                            empty
    run_aggregate_entity_annotations.py    the driver: stream in, stream out, resumable
    utils.py                               all the scoring; opens the LMDB at import (see below)
    refined_lmdb.py                        a copy of ReFinED's LMDB wrapper

## What comes in

A Maverick output record carries annotations from both upstream stages. This step
touches four of its fields and passes the rest of the record through:

| field | produced by | what it contributes |
|---|---|---|
| `hyperlinks_clean` | ReFinED's Wikipedia preprocessing — [`merge_files_and_extract_links.py:111`](../third_party/ReFinED/src/refined/offline_data_generation/merge_files_and_extract_links.py#L111) | anchor spans (`start`, `end`, `surface_form`, `uri`) whose target page resolved to a QID ([`:110`](../third_party/ReFinED/src/refined/offline_data_generation/merge_files_and_extract_links.py#L110)), with list, disambiguation and surname pages already dropped ([`:105-108`](../third_party/ReFinED/src/refined/offline_data_generation/merge_files_and_extract_links.py#L105-L108)) |
| `hyperlinks` | same, [`:93`](../third_party/ReFinED/src/refined/offline_data_generation/merge_files_and_extract_links.py#L93) | *every* anchor the cleaner found, resolved or not. `hyperlinks_clean` is the resolved subset of the same dict objects — [`:109-110`](../third_party/ReFinED/src/refined/offline_data_generation/merge_files_and_extract_links.py#L109-L110) appends by reference and then sets `qcode` on it, so a resolved anchor carries `qcode` in both lists. **Read by nothing here** — `process_doc` only takes `hyperlinks_clean` (`run_aggregate_entity_annotations.py:34`) and then discards both (`:72-73`) |
| `entity_linking` | ReFinED inference — [`run_refined.py:144`](../third_party/ReFinED/run_refined.py#L144) writes it as `spans`; Maverick renames it at [`run_maverick.py:148`](../third_party/maverick-coref/run_maverick.py#L148) | predicted mentions: `start`, `ln`, `text`, `predicted_entity.wikidata_entity_id`, `entity_linking_model_confidence_score` ([`base_types.py:81-108`](../third_party/ReFinED/src/refined/data_types/base_types.py#L81-L108)) |
| `coref` | Maverick — [`run_maverick.py:147`](../third_party/maverick-coref/run_maverick.py#L147) | `clusters_char_offsets` and `clusters_char_text`: document-level coreference clusters, offsets already rebased onto the full document ([`:115-119`](../third_party/maverick-coref/run_maverick.py#L115-L119)) |

Hyperlinks and entity linking give spans that already name an entity.
Coreference gives spans that do not — pronouns, partial names, descriptions —
and the point of this step is to attribute those to a QID as well.

`utils.py:47-76` normalises the first two into one shape (`type`, `start`, `end`,
`text`, `id`, `name`, `confidence`), which is what makes them interchangeable as
candidate anchors for the coref spans. `type` is `"entity"` or `"link"`, and it
decides which list an overlapping candidate joins on a coref span (`:196`,
`:205`) and therefore how it is weighted. Hyperlinks get `confidence: 1.0`
unconditionally (`:74`); entity-linking mentions keep ReFinED's own confidence
(`:59`).

## What goes out

One JSON line per input record: every other field passed through untouched,
`hyperlinks`, `hyperlinks_clean`, `coref` and `entity_linking` popped
(`run_aggregate_entity_annotations.py:72-75`), and `entities` added (`:70`).
Each element of `entities` is built at `utils.py:436-441`:

```json
{
  "char_start": 1204,
  "char_end": 1216,
  "text_mention": "Harry Potter",
  "candidates": [
    {
      "qid": "Q3244512",
      "name": "Harry Potter",
      "scores_by_source": {
        "hyperlinks": 1.0, "entity_linking": 0.98,
        "coref": 0.87, "coref_cluster": 0.62
      },
      "aggregated_score": 0.93
    }
  ]
}
```

| field | meaning |
|---|---|
| `char_start`, `char_end` | character offsets into the record's `text`. They are the join key: evidence from different sources at *identical* offsets merges into one entry, evidence at different offsets stays separate (`utils.py:372-381`) |
| `text_mention` | the span text as the first source to claim those offsets saw it |
| `candidates` | one per QID any source proposed for the span. A span with two plausible entities keeps both |
| `scores_by_source` | the four scores below, `0.0` where that source said nothing |
| `aggregated_score` | the fixed-weight combination below (`0.93` above, not `0.88`: `(4x1.0 + 3x0.98 + 2x0.87 + 1x0.62) / 10`) |

These names are load-bearing downstream. The Elasticsearch mapping declares them
field for field, once per index —
[`retrieval-index/create_es_index.py:69-103`](../retrieval-index/create_es_index.py#L69-L103)
for the case-insensitive index and
[`:140-174`](../retrieval-index/create_es_index.py#L140-L174) for the
case-sensitive one — with `entities` and `entities.candidates` both `nested`, and
`validate_and_normalize_entity_data`
([`:211-286`](../retrieval-index/create_es_index.py#L211-L286)) drops any mention
missing them. Note that the indexer does not read this step's file: it reads the
list back out of the tokenized dataset's per-chunk `metadata`
([`:295`](../retrieval-index/create_es_index.py#L295)). Three further readers use
the same spelling:
[`Untaught/framework/client/es_blacklist.py:195`](../Untaught/framework/client/es_blacklist.py#L195),
[`Ember-on-LMEnt/sentences_gen/blacklist_to_concept_sentences.py:301`](../Ember-on-LMEnt/sentences_gen/blacklist_to_concept_sentences.py#L301)
and [`:541-559`](../Ember-on-LMEnt/sentences_gen/blacklist_to_concept_sentences.py#L541-L559),
and the vendored
[`third_party/OLMo-core/src/examples/kas/dataset_stats.py:77-91`](../third_party/OLMo-core/src/examples/kas/dataset_stats.py#L77-L91).

## The four scores

| source | score |
|---|---|
| `hyperlinks` | always `1.0` — the anchor's fixed confidence (`utils.py:74`) |
| `entity_linking` | ReFinED's model confidence for that mention (`utils.py:59`) |
| `coref` | per-span: `coverage_ratio` for a link, `confidence * coverage_ratio` for an entity-linking mention, taking the max per QID (`utils.py:243-256`); for a span with no overlapping anchor, the best character-level match against the cluster's anchors (`:278`) |
| `coref_cluster` | per-*cluster*: a softmax over the accumulated per-QID weights (`utils.py:292-301`), attached to **every** span in the cluster (`:398-399`) |

`coverage_ratio` (`utils.py:145-166`) is how much of the coref span an
overlapping anchor covers, after stripping leading articles and trailing
possessives so that "the boy's" and "boy" compare cleanly. For spans with no
overlapping anchor at all, `best_entity_or_link_match` (`:113-143`) falls back to
a longest-common-substring F1 against the anchor text — the raw ratio against a
link's `surface_form` (`:123`), the ratio times the anchor's confidence against
an entity-linking anchor's `el_text` (`:133`) — and requires `> 0.3` (`:143`).

Pronoun spans are excluded from scoring (`utils.py:232`), so they get no `coref`
score — but they are still inside the cluster, so they do receive the
`coref_cluster` score. That is the mechanism by which a bare "he" ends up
annotated with a QID.

Clusters are filtered twice before any of this reaches the output: dropped unless
some span overlaps an anchor (`run_aggregate_entity_annotations.py:40-43`), then
dropped unless some span's score reaches `0.3` (`:46-50`).

`aggregated_score` is a weighted mean with **fixed** denominator — weights 4 / 3 /
2 / 1 for hyperlinks / entity linking / coref / coref cluster (`utils.py:364-367`),
divided by their sum, 10, whether or not a source contributed (`:423`), then
clipped and rounded to 2 places (`:425`). So the scale is not what it looks like:
a mention attested only by a hyperlink caps at `0.40`, and `1.00` requires all
four sources at 1.0. This mostly does not matter downstream, because the paper's
thresholds are applied per source against `scores_by_source`, not against
`aggregated_score` — see
[`Untaught/framework/client/es_blacklist.py:39-44`](../Untaught/framework/client/es_blacklist.py#L39-L44)
and [`:195`](../Untaught/framework/client/es_blacklist.py#L195).

One caveat on `coref_cluster`: a softmax over a single candidate is `1.0`, so a
cluster with exactly one candidate QID scores it 1.0 however weak the evidence
was.

## The LMDB

One database, read-only, used only to turn a QID into a human-readable name:

    /home/morg/dataset/refined/organised_data_dir/wikidata_data/qcode_to_wiki.lmdb

It maps QID to Wikidata label. ReFinED builds it during offline preprocessing —
[`build_lmdb_dicts.py:67-68`](../third_party/ReFinED/src/refined/offline_data_generation/build_lmdb_dicts.py#L67-L68)
converts `qcode_to_label.json` into LMDB and
[`:21`](../third_party/ReFinED/src/refined/offline_data_generation/build_lmdb_dicts.py#L21)
puts it under `organised_data_dir/`, at the relative path named in
[`resources_constants.py:97-100`](../third_party/ReFinED/src/refined/constants/resources_constants.py#L97-L100).
The absolute prefix is the same `OUTPUT_DIR` that
[`run_refined.py:18`](../third_party/ReFinED/run_refined.py#L18) writes to and
[`:60`](../third_party/ReFinED/run_refined.py#L60) reads back.

It is consulted in three places: as a fallback when ReFinED supplied no
`human_readable_name` (`utils.py:58`), and for every coref and coref-cluster name
(`:397`, `:399`). Hyperlink names bypass it entirely — they are the Wikipedia
title with underscores replaced by spaces (`:72`). So `name` has mixed
provenance: page titles for hyperlinks, Wikidata labels elsewhere.

`refined_lmdb.py` is ReFinED's
[`lmdb_wrapper.py`](../third_party/ReFinED/src/refined/resource_management/lmdb_wrapper.py)
with `batch_items` inlined and the `ujson`/`tqdm` imports changed, vendored so
that this step does not need ReFinED importable. Only the read path is used here.

### The LMDB is opened at import time

`utils.py:9`, at module level — column 0, not inside a function, not under an
`if __name__` guard:

```python
qcode_to_wiki = LmdbImmutableDict(path="/home/morg/dataset/refined/organised_data_dir/wikidata_data/qcode_to_wiki.lmdb", write_mode=False)
```

`LmdbImmutableDict.__init__` calls `lmdb.open(..., readonly=True, ...)`
immediately (`refined_lmdb.py:67-70`), and the driver does `from utils import *`
at module level too (`run_aggregate_entity_annotations.py:10`). So **importing
either module opens that absolute path before any of this code runs.** On a
machine where the database is not at that path — that is, anywhere but the
cluster the paper was run on — the failure is at import, and `--help` fails with
it. There is no lazy-loading and no fallback; the name lookup is the only thing
the database is used for, but the dependency is unconditional.

## Running it

The driver takes two arguments (`run_aggregate_entity_annotations.py:85-96`):

| argument | meaning |
|---|---|
| `--input_file` | one NDJSON file from `run_maverick.py`, which shards its output as `maverick_<gpu_id>.json` ([`run_maverick.py:316`](../third_party/maverick-coref/run_maverick.py#L316)) |
| `--output_file` | where to write. Append mode (`:59`) |

There is no sharding, no multiprocessing and no directory walk, so "run it on
every file" means one invocation per shard, in parallel if you like. (`csv`,
`glob`, `sys` and `ProcessPoolExecutor` are imported and unused.)

**`--output_file` is not actually honoured.** Line 103 is

```python
output_file = args.output_dir + f"entity_mentions_{args.index}.json"
```

and `parse_args` defines neither `--output_dir` nor `--index`, so the script
raises `AttributeError` there — two statements after parsing its arguments, and
before it opens anything. Reading `args.output_file` instead is the fix; it is a
one-line change, left unmade here because this README does not modify code.

Processing is per document and restartable: `load_processed_ids` (`:24-31`) reads
the ids already in the output file and `:63-64` skips them, and each line is
flushed and `fsync`ed as it is written (`:77-79`), so an interrupted run resumes
by being re-run with the same arguments. Three page ids are skipped
unconditionally at `:66` — `44471088`, `62180015`, `67215817` — with no comment
explaining why.

Dependencies are `lmdb`, `ujson`, `numpy` and `tqdm`, all pinned in the repo's
[`environment.yml`](../environment.yml) (lines 232, 388, 261, 374). No GPU.

## What an outside reader must change

| where | value | what to do |
|---|---|---|
| `utils.py:9` | `/home/morg/dataset/refined/organised_data_dir/wikidata_data/qcode_to_wiki.lmdb` | point at your own ReFinED `organised_data_dir`, and move the open inside a function if you want the module importable without it |
| `run_aggregate_entity_annotations.py:103` | `args.output_dir` / `args.index` | use `args.output_file`; as written the entry point cannot run |
| `run_aggregate_entity_annotations.py:66` | three hardcoded page ids | remove unless you are reproducing the paper's corpus exactly |

`:102` also carries `/home/morg/dataset/maverick/maverick_0.json` in a trailing
comment. Those two are the only absolute paths in the component.

## Rough edges

- **Dead code.** `aggregate_mention_entity_scores` (`utils.py:309-357`) is never
  called from anywhere in the repo, and its source key is `"coref-cluster"`
  where the live path uses `coref_cluster` — it is a stale earlier version of
  `aggregate_mentions`. `total_contribution` is accumulated and clamped
  (`:290`) and then never read. `process_doc`'s `skip_coref` branch (`:51-53`)
  is unreachable from the driver.
- **Missing input fields crash rather than degrade.** `process_doc` uses `.get`
  with `[]` defaults (`:34-36`), but `enrich_coref_clusters` subscripts
  `coref["clusters_char_offsets"]` (`utils.py:172`), so a record without a
  `coref` field raises `TypeError`, and the four `pop`s at `:72-75` have no
  default, so a record missing any of them raises `KeyError`. Every record the
  normal pipeline produces has all four.
- **`refined_lmdb.py`'s write path is broken in this copy.** Line 9 is
  `import tqdm` (the module) while `from_dict` calls `tqdm(...)` at `:125`.
  Unused here — this step only reads — but do not reach for `from_dict`.
- **The vendored dolma tokenizer does not read this schema as shipped.**
  `--output_file`'s help text says the output is ingested by the dolma
  tokenizer, and the tokenizer does read `char_start`/`char_end`
  ([`kas_tokenizer.py:58-59`](../third_party/dolma/python/dolma/tokenizer/kas_tokenizer.py#L58-L59)),
  but its input struct declares no `entities` field
  ([`data_types.py:19-26`](../third_party/dolma/python/dolma/core/data_types.py#L19-L26)),
  it subscripts the decoded struct at
  [`:106`](../third_party/dolma/python/dolma/tokenizer/kas_tokenizer.py#L106),
  it asserts against `entity["text"]` at
  [`:79`](../third_party/dolma/python/dolma/tokenizer/kas_tokenizer.py#L79)
  where this step emits `text_mention`, and the
  [`kas.yaml`](../third_party/dolma/python/dolma/configs/kas.yaml) that
  [`run.slurm:17`](../third_party/dolma/python/run.slurm#L17) passes to the CLI
  globs the Maverick output directory rather than this step's output.
  Expect to reconcile those before step 4 runs on a fresh corpus.
