# retrieval-index — the LMEnt Elasticsearch index

One script, [`create_es_index.py`](create_es_index.py), builds the entity-based
retrieval index: one Elasticsearch document per pretraining chunk, carrying the
chunk's decoded text and every entity mention the annotation pipeline found in
it. It is what makes *"which chunks mention Q8337?"* and *"which chunks contain
this phrase?"* answerable — and, because a document's `chunk_id` is the dataset
instance index, what lets either answer be traced back to the training step that
consumed the chunk.

Building from scratch takes up to a day, and the script needs edits before it
runs anywhere (see *Building the index from scratch*). Both finished indexes are
also downloadable as a 127GB Elasticsearch snapshot; the restore procedure is in
the [root README](../README.md), under *Entity-Based Retrieval Index*. This file
documents what is inside them and how to query them.

---

## The two indexes

The script writes **one** index per run, chosen by the `index_name` literal at
[line 336](create_es_index.py#L336); [line 339](create_es_index.py#L339) picks
the mapping from that same name. Building both means running it twice.

| `index_name` in the script | name after the snapshot restore | analyzer for text fields | `text.raw` |
|---|---|---|---|
| `enwiki_case_sensitive` | `lment_cs` | custom `default`: `standard` tokenizer, **empty** filter list ([120-128](create_es_index.py#L120-L128)) — tokens keep their original case | not mapped |
| `enwiki` | `lment_ci` | no `analysis` block ([43-53](create_es_index.py#L43-L53)), so Elasticsearch's built-in default `standard` analyzer applies, which lowercases | `keyword` ([63-65](create_es_index.py#L63-L65)) |

Everything else is shared: one shard, no replicas, `BM25` set explicitly as the
default similarity, and `"dynamic": False` so only the mapped fields below are
searchable ([44-55](create_es_index.py#L44-L55),
[111-131](create_es_index.py#L111-L131)).

Two consequences of how case-sensitivity is implemented here:

- The custom analyzer is declared as the index's **`default`**, so it applies at
  index time *and* at search time, and it applies to every analyzed field in
  that index — `title` and `entities.text_mention` and
  `entities.candidates.name`, not just `text`. Resolving an entity *by name*
  against `lment_cs` is therefore also case-sensitive.
- Neither index stems or removes stopwords. Matching is on whole tokens, exactly
  as written in `lment_cs` and case-folded in `lment_ci`.

Which one a caller wants:

| Task | Index |
|---|---|
| Entity retrieval by QID | Either. `entities.candidates.qid` is a `keyword` in both mappings ([83](create_es_index.py#L83), [154](create_es_index.py#L154)), so no analyzer touches it and the choice of index does not change QID matching. |
| String retrieval where case carries meaning — acronyms, proper names, `Apple` vs `apple` | `lment_cs` |
| String retrieval that should find every capitalization of a surface form | `lment_ci` |

`Untaught` defaults to the case-sensitive index for exactly the first reason:
see [`es_blacklist.py:49-59`](../Untaught/framework/client/es_blacklist.py#L49-L59)
and `ES_INDEX=lment_cs` in
[`Untaught/framework/env.sh:46`](../Untaught/framework/env.sh#L46).

---

## What a document looks like

Each document is built in
[`fetch_and_prepare`](create_es_index.py#L288-L313) from one dataset instance:
`DATASET[idx]` gives the token ids and the per-chunk metadata, and the body is
assembled at [298-307](create_es_index.py#L298-L307).

| Field | Type | Meaning |
|---|---|---|
| `chunk_id` | `integer` | The dataset instance index — see the invariant below. Also used as the document `_id` ([430](create_es_index.py#L430)), so `GET /<index>/_doc/<chunk_id>` works. |
| `article_id` | `integer` | The `id` column of the tokenization metadata: the source document this chunk was cut from. |
| `title` | `text` | The source document's title. |
| `metadata_source` | `text` | The metadata `src` column, carried through from tokenization ([302](create_es_index.py#L302)). Its value set is fixed by the tokenization step and is not defined anywhere in this repo. |
| `text` | `text` | The chunk itself, decoded from `input_ids` ([303](create_es_index.py#L303)) with the `allenai/dolma2-tokenizer` ([`tokenizer.py:17`](../third_party/OLMo-core/src/olmo_core/data/tokenizer.py#L17)). This is the field to search for strings. |
| `start`, `end` | `integer` | The token span of the **source document** in its tokenized `.npy` array — not the chunk's span. Metadata is per-document, and the dataset returns the record whose `[start, end]` contains the chunk ([`numpy_dataset.py:1571`](../third_party/OLMo-core/src/olmo_core/data/numpy_dataset.py#L1571), used as the document's own origin at [`:1580`](../third_party/OLMo-core/src/olmo_core/data/numpy_dataset.py#L1580)). |
| `entities` | `nested[]` | The mentions that fall inside this chunk, filtered from the document's mention list ([`numpy_dataset.py:1509-1512`](../third_party/OLMo-core/src/olmo_core/data/numpy_dataset.py#L1509-L1512)). |
| `entities.char_start`, `entities.char_end` | `integer` | Half-open character span of the mention **within `text`**. The dataset rebases document offsets onto the chunk ([`numpy_dataset.py:1514-1518`](../third_party/OLMo-core/src/olmo_core/data/numpy_dataset.py#L1514-L1518)), so these index directly into the string you get back. |
| `entities.text_mention` | `text` + `.raw` `keyword` | The mention's surface string. |
| `entities.candidates` | `nested[]` | Candidate entities for that mention. A mention with no candidates is dropped entirely ([283-284](create_es_index.py#L283-L284)). |
| `entities.candidates.qid` | `keyword` | Wikidata QID. |
| `entities.candidates.name` | `text` + `.raw` `keyword` | Candidate label; `""` when the pipeline had none ([`utils.py:430`](../aggregate_entity_annotations/utils.py#L430)). |
| `entities.candidates.aggregated_score` | `float` | The four source scores weighted 4 / 3 / 2 / 1 ([`utils.py:364-367`](../aggregate_entity_annotations/utils.py#L364-L367)), divided by the total weight, clipped to `[0, 1]` and rounded to 2dp ([`utils.py:417-425`](../aggregate_entity_annotations/utils.py#L417-L425)). |
| `entities.candidates.scores_by_source.{hyperlinks, entity_linking, coref, coref_cluster}` | `float` | Per-source confidence, `0.0` where that source said nothing ([`utils.py:411-414`](../aggregate_entity_annotations/utils.py#L411-L414)). |

The four sources are the four annotation signals merged by
[`aggregate_mentions`](../aggregate_entity_annotations/utils.py#L359):
Wikipedia hyperlinks, ReFinED entity linking, Maverick coreference, and the
cluster-level entity attribution derived from it. `validate_and_normalize_entity_data`
([211-286](create_es_index.py#L211-L286)) re-checks the mention fields and
coerces every score to `float` before indexing, dropping what does not conform;
it does not validate `name`, which is passed through as the pipeline left it.

Abridged, and with every value except `chunk_id` invented for illustration, a
document reads:

```json
{
  "_id": "2955255",
  "_source": {
    "chunk_id": 2955255,
    "article_id": 2698,
    "title": "Harry Potter",
    "metadata_source": "...",
    "text": "... Harry Potter is a series of seven fantasy novels ...",
    "start": 141238272,
    "end": 141241088,
    "entities": [
      {
        "char_start": 4,
        "char_end": 16,
        "text_mention": "Harry Potter",
        "candidates": [
          {
            "qid": "Q8337",
            "name": "Harry Potter",
            "aggregated_score": 0.68,
            "scores_by_source": {
              "hyperlinks": 1.0,
              "entity_linking": 0.93,
              "coref": 0.0,
              "coref_cluster": 0.0
            }
          }
        ]
      }
    ]
  }
}
```

`(4 x 1.0 + 3 x 0.93 + 2 x 0.0 + 1 x 0.0) / 10 = 0.68` — the aggregated score is
always recomputable from `scores_by_source`.

---

## The `chunk_id` invariant

> **A document's `chunk_id` is the index of that chunk in the OLMo-core dataset
> the trainer iterates.**

This is the single property the rest of the suite is built on. Three of its four
parts hold by construction; the fourth is a deployment fact, not a code fact.

| Claim | Where |
|---|---|
| The stored id *is* the loop index | [`create_es_index.py:299`](create_es_index.py#L299) — `'chunk_id': idx`, where `idx` is the argument `DATASET[idx]` was called with ([294](create_es_index.py#L294)) |
| The dataset is *configured* like the trainer's | [`create_es_index.py:322-331`](create_es_index.py#L322-L331) builds a `NumpyDatasetConfig.glob` whose curriculum and tokenizer parameters match [`train.py:203-212`](../third_party/OLMo-core/src/examples/kas/train.py#L203-L212) as configured in [`kas_config.json`](../third_party/OLMo-core/src/examples/kas/kas_config.json): `kas_vsl`, `max_sequence_length` 2048, `min_sequence_length` 64, `grow_p2` over 8 cycles unbalanced, the dolma2 tokenizer ([`train.py:111`](../third_party/OLMo-core/src/examples/kas/train.py#L111)), `include_instance_metadata` |
| …but **not** pointed at the data by the same means | The `.npy` glob and the `work_dir` are separate hard-coded literals on each side ([317](create_es_index.py#L317), [323](create_es_index.py#L323) here; the `dataset` block of `kas_config.json` there), and they do not currently name the same directories. Index assignment depends on *which* shards are globbed, so this is the part that has to be checked per deployment rather than read off the code |
| File order within the glob is deterministic | [`NumpyDatasetConfig.glob`](../third_party/OLMo-core/src/olmo_core/data/numpy_dataset.py#L1847) sets `expand_glob=True`, and expansion is `matches = sorted(glob(glob_path))` ([`numpy_dataset.py:1902`](../third_party/OLMo-core/src/olmo_core/data/numpy_dataset.py#L1902)), so index assignment does not depend on filesystem order |
| Training batches carry the same id | [`data_loader.py:466`](../third_party/OLMo-core/src/olmo_core/data/data_loader.py#L466) — `return dict(**item, index=idx)` |

What it buys:

- **Retrieval hit to training step.** `batch_indices.npy` records the chunk ids
  in each step's batch, so for a model trained one epoch a `chunk_id` from a
  query resolves to the step that consumed it. The loop is in the root README's
  *Analyzing Learning Dynamics* section; the file is written by
  [`write_dataloader_batch_indices.py`](../third_party/OLMo-core/src/examples/kas/write_dataloader_batch_indices.py).
- **Retrieval hit to held-out chunk.** `Untaught` resolves QIDs to chunk ids on
  the login node and masks the matching rows during training; see
  [`Untaught/README.md`](../Untaught/README.md).

Because the last-but-two row above is a deployment fact, a stale snapshot or a
missing tokenized shard breaks the invariant silently.
[`Untaught/tests/verify_chunk_alignment.py`](../Untaught/tests/verify_chunk_alignment.py)
checks it empirically: it compares the index's document count against the
dataset length, then samples random dataset indices, decodes each chunk, and
requires the string to equal the `text` stored under that `chunk_id`.

```sh
cd "${LMENT_ROOT}/Untaught" && . ./activate_env.sh
python tests/verify_chunk_alignment.py --config configs/train_170m_control.yaml -n 25
```

Exit code 0 means aligned. Run it once against any deployment before trusting
either mapping above.

---

## Querying the index

The examples below assume the connection variables that
[`Untaught/framework/env.sh`](../Untaught/framework/env.sh#L43-L51) exports —
`ES_SCHEME`, `ES_HOST`, `ES_PORT`, `ES_INDEX`, `ES_PASSWORD` — and the `lment`
conda environment, which pins `elasticsearch==8.18.1`
([`environment.yml:154`](../environment.yml#L154)). Certificate verification is
off on both sides of this deployment, hence `verify_certs=False` and curl's `-k`.

```python
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

es = Elasticsearch(
    f"{os.environ.get('ES_SCHEME', 'https')}://"
    f"{os.environ.get('ES_HOST', 'localhost')}:{os.environ.get('ES_PORT', 9200)}",
    basic_auth=("elastic", os.environ.get("ES_PASSWORD", "")),
    request_timeout=300,
    max_retries=10,
    retry_on_timeout=True,
    verify_certs=False,
    ssl_show_warn=False,
)
INDEX = os.environ.get("ES_INDEX", "lment_cs")
```

That is the same client
[`es_blacklist.get_esclient`](../Untaught/framework/client/es_blacklist.py#L65-L94)
builds; import it instead of copying this if you are working inside `Untaught`.
The `body=` form used below is what `es_blacklist.py` uses against the pinned
8.18.1 client.

### Entity-based retrieval: every chunk that mentions a QID

`entities` is nested and `entities.candidates` is nested inside it, so a mention
query needs two levels of `nested`. The QID clause is a `terms` filter on a
`keyword` field; the score clauses are a `should` with
`minimum_should_match: 1`, so a mention qualifies when **at least one** source
is confident enough. The thresholds are the paper's, as recorded in
[`es_blacklist.py:39-44`](../Untaught/framework/client/es_blacklist.py#L39-L44).

```python
QIDS = ["Q8337"]                       # Harry Potter (the series)
THRESHOLDS = {
    "hyperlinks": 1.0,
    "entity_linking": 0.6,
    "coref": 0.6,
    "coref_cluster": 0.6,
}

entity_query = {
    "nested": {
        "path": "entities",
        "query": {
            "nested": {
                "path": "entities.candidates",
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"entities.candidates.qid": QIDS}},
                            {
                                "bool": {
                                    "should": [
                                        {"range": {f"entities.candidates.scores_by_source.{src}": {"gte": v}}}
                                        for src, v in THRESHOLDS.items()
                                    ],
                                    "minimum_should_match": 1,
                                }
                            },
                        ]
                    }
                },
            }
        },
    }
}

print(es.count(index=INDEX, body={"query": entity_query})["count"])

chunk_ids = sorted(
    hit["_source"]["chunk_id"]
    for hit in scan(
        es,
        index=INDEX,
        query={"query": entity_query, "_source": ["chunk_id"]},
        size=5000,
        preserve_order=False,
    )
)
```

That query body is exactly what
[`build_entity_query`](../Untaught/framework/client/es_blacklist.py#L185-L216)
produces, and the `scan` call is
[`fetch_chunk_ids`](../Untaught/framework/client/es_blacklist.py#L152-L182).
`scan` is the right helper: a broad concept matches more chunks than a single
`_search` response returns (10,000 hits by default). Drop the two score clauses
to get every chunk with *any* candidate for the QID, at any confidence.

Because the whole query is a `filter`, hits are unscored — fine for retrieval,
but add a `match` clause or a `sort` if you want an order. To see *which* mention
matched, put `"inner_hits": {}` on the outer `nested` clause; Elasticsearch then
returns the matching `entities` sub-documents, each with its `char_start`,
`char_end`, `text_mention` and `candidates`.

The same query with curl, counting instead of scrolling:

```bash
curl -k -u "elastic:$ES_PASSWORD" -H 'Content-Type: application/json' \
  -X POST "https://$ES_HOST:$ES_PORT/lment_cs/_count?pretty" -d '{
  "query": {
    "nested": {
      "path": "entities",
      "query": {
        "nested": {
          "path": "entities.candidates",
          "query": {
            "bool": {
              "filter": [
                {"term": {"entities.candidates.qid": "Q8337"}},
                {"bool": {
                  "minimum_should_match": 1,
                  "should": [
                    {"range": {"entities.candidates.scores_by_source.hyperlinks": {"gte": 1.0}}},
                    {"range": {"entities.candidates.scores_by_source.entity_linking": {"gte": 0.6}}},
                    {"range": {"entities.candidates.scores_by_source.coref": {"gte": 0.6}}},
                    {"range": {"entities.candidates.scores_by_source.coref_cluster": {"gte": 0.6}}}
                  ]
                }}
              ]
            }
          }
        }
      }
    }
  }
}'
```

To go the other way — surface name to QID — aggregate on
`entities.candidates.qid` under a `match_phrase` on `entities.candidates.name`;
`es_blacklist.py`'s `resolve` command
([lines 300-364](../Untaught/framework/client/es_blacklist.py#L300-L364)) is
that query, with mention counts.

### String-based retrieval: every chunk containing a phrase

Plain `match_phrase` on `text`. This is where the index choice matters: against
`lment_cs` only this capitalization matches, against `lment_ci` any
capitalization does.

```python
string_query = {"match_phrase": {"text": "Hermione Granger"}}

resp = es.search(
    index="lment_cs",
    body={
        "size": 3,
        "query": string_query,
        "_source": ["chunk_id", "article_id", "title"],
        "highlight": {"fields": {"text": {}}},
    },
)
for hit in resp["hits"]["hits"]:
    src = hit["_source"]
    print(src["chunk_id"], src["title"], hit["highlight"]["text"][0][:160])
```

`highlight` is stock Elasticsearch highlighting on an analyzed field; it needs
nothing extra in the mapping. Use `match` rather than `match_phrase` for a
BM25-ranked bag-of-tokens search — both mappings set `BM25` as the default
similarity explicitly.

```bash
curl -k -u "elastic:$ES_PASSWORD" -H 'Content-Type: application/json' \
  -X POST "https://$ES_HOST:$ES_PORT/lment_ci/_search?pretty" -d '{
  "size": 3,
  "query": {"match_phrase": {"text": "hermione granger"}},
  "_source": ["chunk_id", "title"]
}'
```

Run that against `lment_cs` and it returns nothing, because the case-sensitive
analyzer indexed `Hermione`, not `hermione`. That contrast is the whole reason
both indexes exist.

`text.raw` — present only in `lment_ci` — is the entire chunk as one
unanalyzed `keyword` term. It is for exact whole-chunk equality, e.g.
`{"term": {"text.raw": "<the full chunk string>"}}`, not for substring or
phrase search.

### Both at once, and back to a training step

A co-occurrence query is the two above under one `bool`:

```python
resp = es.search(
    index="lment_cs",
    body={
        "size": 5,
        "query": {"bool": {"filter": [entity_query, {"match_phrase": {"text": "Philosopher's Stone"}}]}},
        "_source": ["chunk_id", "title"],
    },
)
```

A chunk id can be read straight back as a document, because the id is the
document `_id`:

```bash
curl -k -u "elastic:$ES_PASSWORD" "https://$ES_HOST:$ES_PORT/lment_cs/_doc/2955255?pretty"
```

and, for a model trained one epoch, turned into the step that saw it with
`LMEnt-Dataset/dataset-cache/batch_indices.npy` — the loop in the root README's
*Analyzing Learning Dynamics* section, which reports `chunk_id` 2955255 at step
16618. Retrieval and learning dynamics meet at this integer.

---

## Building the index from scratch

**The script needs one source edit before it runs anywhere.** The `work_dir` and
the `.npy` glob are hard-coded to absolute `/home/morg/...` paths
([317](create_es_index.py#L317), [323](create_es_index.py#L323)) and must be
repointed at your own unpacked `LMEnt-Dataset`; see *Rough edges*. Restoring the
published snapshot is the supported path, and this section is for anyone
rebuilding from scratch.

The script needs, all at once:

- **The `lment` conda environment.** `elasticsearch==8.18.1` and
  `ai2-olmo-eval==0.5.0` — the latter provides the `olmo_eval.HFTokenizer`
  imported in each worker ([194-205](create_es_index.py#L194-L205)) — are both in
  [`environment.yml`](../environment.yml). The tokenizer itself,
  `allenai/dolma2-tokenizer`, is fetched from Hugging Face, so a warm HF cache
  matters when 128 workers start at once.
- **OLMo-core on `sys.path`.** Lines [21-30](create_es_index.py#L21-L30) take
  `OLMO_CORE_SRC` if it is set, otherwise derive
  `<repo root>/third_party/OLMo-core/src` from the script's own location, and
  raise `SystemExit` if whichever path was chosen is not a directory.
  `OLMO_CORE_SRC` is one of the variables
  [`Untaught/framework/env.sh`](../Untaught/framework/env.sh#L33) sets, so
  sourcing [`Untaught/activate_env.sh`](../Untaught/activate_env.sh) or
  [`Ember-on-LMEnt/activate_env.sh`](../Ember-on-LMEnt/activate_env.sh) is
  enough. The path has to be importable as a package tree — line 32 imports
  `examples.kas.train`, which pulls in `olmo_core` and torch.
- **A reachable Elasticsearch.** `Untaught/activate_env.sh` starts one on the
  login node if nothing is responding there and `ES_HOME` is yours to start. Set
  `http.max_content_length: 1GB` in `elasticsearch.yml` first — the note at
  [line 15](create_es_index.py#L15); the bulk batches are large.
- **An unpacked `LMEnt-Dataset` that [`setup.sh`](../setup.sh) has been run
  over.** The `.npy` token arrays ship with the download; `setup.sh` gunzips the
  per-chunk metadata CSVs into `dataset-cache/dataset-metadata/` and creates the
  `part-N` symlinks the dataloader looks for, in `dataset-metadata/` and
  `dataset-common/`. The script then wants `dataset-tokenized/*.npy` for its glob
  and `dataset-cache/` as its `work_dir`.

Then, noting that `activate_env.sh` leaves the shell in `Untaught/`:

```sh
. Untaught/activate_env.sh          # OLMO_CORE_SRC, PYTHONPATH, conda, Elasticsearch
python "${LMENT_ROOT}/retrieval-index/create_es_index.py"
```

The script refuses to touch an index that already exists
([341-343](create_es_index.py#L341-L343)), creates it with the chosen mapping,
raises `index.mapping.nested_objects.limit` to 10^8
([349-353](create_es_index.py#L349-L353)) because one chunk can carry thousands
of nested mention and candidate sub-documents, and then fans out: a
128-process pool decodes chunks and builds bodies
([400-411](create_es_index.py#L400-L411)), four threads ship them in batches of
250 with five exponential-backoff retries
([359-384](create_es_index.py#L359-L384)). The root README budgets **up to a
day** for the full corpus; one likely contributor is the synchronous `es.exists`
round trip per chunk on the main loop ([427](create_es_index.py#L427)).

---

## Rough edges

Things an outside reader will hit. None of them are hidden in the code, but none
of them are guarded either.

| Where | What |
|---|---|
| [317](create_es_index.py#L317), [323](create_es_index.py#L323) | **Absolute cluster paths.** The `work_dir` and the `.npy` glob are hard-coded to `/home/morg/...` directories that exist only on the machine this was written on. Both must be repointed at your own unpacked `LMEnt-Dataset` — `dataset-cache` and `dataset-tokenized/*.npy` — before the script can run anywhere else, and at the *same* shards the trainer uses, or the `chunk_id` invariant does not hold. Nothing reads them from the environment or the command line. |
| [336](create_es_index.py#L336) | **The index name is a literal**, and [339](create_es_index.py#L339) derives the mapping from it by string comparison: any name other than `enwiki_case_sensitive` silently gets the case-*insensitive* mapping. Edit both together, or not at all. |
| [335](create_es_index.py#L335), [180-190](create_es_index.py#L180-L190) | **Connection details are not read from the environment.** `get_esclient()` is called with no arguments, so it connects to `https://localhost:9200` as `elastic` with an **empty** password. The `ES_*` variables are used by the query-side code, not here; supply credentials at this call site. |
| [341-343](create_es_index.py#L341-L343) vs [427](create_es_index.py#L427) | **Resume is half-implemented.** The per-document `es.exists` check and the `start` offset at [393](create_es_index.py#L393) exist to let a run be resumed, but the index-exists guard returns before either can be used. Resuming means removing that guard deliberately. |
| [253-255](create_es_index.py#L253-L255) | The guard meant to drop candidates with no QID reads `if not qid and isinstance(qid, str)`, which is true only for the empty string; a `None` QID passes validation and is indexed as a null, leaving that candidate unsearchable by `qid` while still present in `_source`. |
| [251](create_es_index.py#L251), [276](create_es_index.py#L276) | `name` is the one candidate field never type-checked — it is read and written through unchanged, so whatever the annotation pipeline produced is what gets indexed. |

Several imports are unused: `random`, `torch`, `tqdm`, the `threading` module
itself (only `from threading import Thread` is used), `AutoTokenizer`
([33](create_es_index.py#L33)), and `build_config`, `seed_all`,
`set_random_seeds` ([32](create_es_index.py#L32)). The `examples.kas.train`
import still has to *resolve*, which is what makes the whole OLMo-core and torch
stack a hard dependency even though none of those three names is called. The
rest of `olmo_core` is genuinely used, at
[320-331](create_es_index.py#L320-L331).
