# Untaught — train an LMEnt model without a concept

Hold specific entities out of an LMEnt pretraining run, using the Elasticsearch
index that already ships with the suite. **Nothing here rebuilds the dataset,
re-tokenizes anything, or touches the index.** Both are used read-only.

---

## The mechanism

One fact makes this simple:

> **The `chunk_id` in the Elasticsearch index is the dataset instance index,
> and every training batch carries those same ids in `batch["index"]`.**

So excluding a concept is three steps:

```
  ES: "which chunks mention Q8337?"  ->  [1204, 88931, ...]      (chunk ids)
                                              |
  training: batch["index"] = [1204, 55, 9012, ...]               (same ids)
                                              |
  set batch["instance_mask"] = False for the matching rows       (masked)
```

A masked row's labels all become `-100`, so that chunk contributes **zero
gradient**. The model reads it in the forward pass but learns nothing from it.

### Why this is true, not assumed

| Claim | Where it is proven |
|---|---|
| ES `chunk_id` is the dataset index | [`create_es_index.py:286`](../retrieval-index/create_es_index.py#L286) stores `'chunk_id': idx` while iterating a `NumpyDatasetConfig` **identical** to the trainer's ([lines 302-320](../retrieval-index/create_es_index.py#L302-L320) vs [`train.py:203-212`](../OLMo-core/src/examples/kas/train.py#L203-L212)) |
| Batches carry that id | [`data_loader.py:466`](../OLMo-core/src/olmo_core/data/data_loader.py#L466) — `return dict(**item, index=idx)` |
| `instance_mask` kills the loss for a row | [`data/utils.py:576-577`](../OLMo-core/src/olmo_core/data/utils.py#L576-L577) — `labels.masked_fill_(~instance_mask.unsqueeze(-1), label_ignore_index)` |
| OLMo-core supports this natively | [`trainer.py:1173-1175`](../OLMo-core/src/olmo_core/train/trainer.py#L1173-L1175) already logs a `train/masked instances` metric when an `instance_mask` is present |
| `pre_step` can mutate the batch in time | [`trainer.py:1325`](../OLMo-core/src/olmo_core/train/trainer.py#L1325) calls `pre_step(batch)`, [`:1327`](../OLMo-core/src/olmo_core/train/trainer.py#L1327) then calls `_train_batch(batch)`, which builds labels at [`:1185`](../OLMo-core/src/olmo_core/train/trainer.py#L1185) |
| The mask survives micro-batching | [`data/utils.py:43-45`](../OLMo-core/src/olmo_core/data/utils.py#L43-L45) — `split_batch` splits every tensor key along dim 0 |

This is the LMEnt paper's §3.2 capability — *"retrieve all chunks that mention
certain entities"* — pointed at training instead of analysis.

### Why mask instead of delete

VSL batches are **token-constant, instance-variable**:
`instances_per_batch = global_batch_size // bucket_seq_len`, and
[`_batch_index_to_local_instance_indices`](../OLMo-core/src/olmo_core/data/data_loader.py#L763-L788)
slices them **per rank**. Physically dropping instances would give different
ranks different instance counts — desynchronising FSDP collectives — and would
change the effective batch size versus the control.

Masking keeps batch shapes, step counts and data order **identical** between the
control and ablated runs. The only difference is the gradient. That is exactly
what a controlled comparison needs.

---

## Layout

```
Untaught/
├── untaught/
│   ├── exclusion.py         ChunkExclusionCallback — the ~15 lines that matter
│   ├── es_blacklist.py      entity QIDs -> chunk-id .npy, via Elasticsearch
│   ├── config_env.py        ${VAR} expansion for configs; sources env.sh on miss
│   └── train_untaught.py    wraps examples/kas/train.py, attaches the callback
├── configs/
│   ├── env.sh               paths, ES connection, SLURM defaults
│   ├── entities/harry_potter.json
│   ├── train_170m_control.json
│   └── train_170m_no_harry_potter.json
├── slurm/
│   ├── train.slurm
│   └── build_blacklist.slurm
├── tests/
│   ├── test_exclusion.py    proves masked chunks leave the loss (upstream semantics)
│   ├── test_untaught_units.py  unit tests: callback, ES query, config expansion
│   └── verify_chunk_alignment.py  REMOTE preflight: proves chunk_id alignment empirically
├── run_smoke_control.sh     smoke 1: 170M, nothing excluded
└── run_smoke_harry_potter.sh smoke 2: 170M, "Harry Potter" excluded
```

Upstream code is **not modified**. `train_untaught.py` imports `build_config`
from `examples/kas/train.py` verbatim, so the model, optimizer, dataset, VSL
curriculum and data order stay exactly upstream.

---

## Setup

On the TAU cluster (`ssh user@slurm-client.cs.tau.ac.il`):

```sh
cd $LMENT_ROOT/Untaught
chmod +x run_smoke_*.sh

# adjust if your paths differ from /home/morg/NLP_2526b/stahli;
# check your account/partition with:  sacctmgr -P -i show user -s "$USER"
vim configs/env.sh

export ES_PASSWORD='...'          # never commit this
conda activate lment
```

Preflight, in order, all GPU-free:

```sh
# 1. unit + semantics tests (seconds)
python tests/test_exclusion.py
python tests/test_untaught_units.py

# 2. config builds, paths resolve, blacklist readable (seconds)
./run_smoke_control.sh --check

# 3. THE decisive check — proves on this deployment that ES chunk_id
#    equals the dataset instance index, by comparing decoded chunk text
#    against the indexed text for random chunks (a few minutes; run once)
. configs/env.sh                  # only ES_PASSWORD really needs your shell:
                                  # the scripts source env.sh themselves if a
                                  # config path comes out unresolved
python tests/verify_chunk_alignment.py --config configs/train_170m_control.json -n 25
```

If step 3 fails, **do not train** — the blacklist would exclude the wrong
chunks. It means the deployed index and dataset directory disagree (stale
snapshot, missing/renamed tokenized file).

---

## The two smoke runs

### 1 — control (nothing excluded)

```sh
./run_smoke_control.sh
```

170M, 200 steps, no blacklist. This is the baseline and it also proves the
plumbing works before Elasticsearch is involved at all.

### 2 — "Harry Potter" excluded

```sh
./run_smoke_harry_potter.sh --resolve   # which QIDs does the corpus use?
./run_smoke_harry_potter.sh --count     # how many chunks would go?
./run_smoke_harry_potter.sh             # build blacklist + submit
```

Step 1 queries the index and writes `blacklists/harry_potter.npy`. Step 2 trains
with those chunks masked.

**Verify the QIDs before trusting a run.** `--resolve` reports which QIDs the
corpus actually attaches to that name, with mention counts. The defaults in
`configs/entities/harry_potter.json` are `Q8337` (the series) and `Q3244512`
(the character); a franchise, its characters and its individual books are all
separate Wikidata entities, so decide how wide your concept is.

### What to look for

| Signal | Control | Ablated |
|---|---|---|
| `blacklist` line in the header | `(none) -- CONTROL run` | path + chunk count |
| `train/untaught excluded instances` | absent | non-zero on some steps |
| `train/masked instances` (OLMo-core's own) | absent | matches the above |
| `train/untaught guard leaks` | absent | **must stay absent** (see caveats) |
| total steps | 200 | 200 (identical — that is the point) |

If the ablated run shows all zeros, the blacklist did not match anything —
check your QIDs with `--resolve`.

---

## Going from smoke to a real run

In both configs, change:

```json
"max_duration": { "value": 1, "unit": "epochs" }
```

One epoch is ~109K steps (3.6B tokens ÷ 32,768 tokens/step). For a 170M model
that is roughly 4–8 hours on one H100-class GPU. Then also:

- `"disable_downstream_eval": false` and set `eval_interval` to `1000`
- raise `checkpointer.save_interval` to `1000` (the paper's cadence, 110
  checkpoints/epoch)
- bump `UNTAUGHT_TIME` (minutes); `studentbatch` allows 3 days, 6 jobs max

The configs already use the paper's hyperparameters (appendix B.4): AdamW,
global batch 32,768 tokens, rank batch 8,192, peak LR 5e-4, weight decay 0.05,
1,000 warmup steps. **These differ from the checked-in
`OLMo-core/src/examples/kas/kas_config.json`**, which has LR 3e-4 / WD 0.01 and
does not match the released models.

---

## Honest caveats

**Exclusion is not erasure.** You remove chunks the annotations *flag*. The
paper's own error analysis (§B.3) found ~2.7% annotation errors, and an entity
can be described without being named by hyperlinks, entity linking or coref. Say
"sharply reduced exposure", not "never saw it".

**The ablated run sees slightly fewer effective tokens.** Masked chunks still
occupy batch slots. For one entity (a few thousand chunks out of 10.5M, ~0.02%)
this is negligible. For a large blacklist it is not, and you would want a
random-ablation control — drop a random set of chunks matched in count and
length distribution — to separate "this concept" from "less data".

**Thresholds are a recall/precision dial.** Defaults are the paper's validated
values (H=1, EL=C=CC=0.6, §5.2/Table 4). Lower them to catch more mentions at
the cost of collateral chunks; raise them for high-confidence mentions only.

**One guard exists for a reason.** `_train_batch` divides the summed loss by the
number of live label tokens. If a rank masked its *entire* batch that divisor is
zero and the loss goes NaN, poisoning the all-reduce for every rank.
`guard_all_masked` keeps one instance to prevent it — **which leaks one
blacklisted chunk into training**, so every firing is recorded under
`train/untaught guard leaks`. It should never fire with a single-entity
blacklist (the probability of 16–512 uniformly-shuffled instances all being
blacklisted at ~0.1% rate is astronomically small); if it fires at all, audit
the run, and if it fires often, chunk-level exclusion is the wrong tool for
that blacklist size.

---

## Notes

- `include_instance_metadata` is `false` in both configs, on purpose. Exclusion
  keys off `batch["index"]`, never off per-chunk entity metadata, so training
  never reads the 212GB `dataset-metadata` CSVs. (The dataset constructor still
  memory-loads the small `metadata-part-N-00000.npy` line-offset indexes that
  `setup.sh` symlinked — those must exist, and on this remote they do.) This
  matches OLMo-core's own advice in `write_dataloader_batch_indices.py`.
- **GPU constraint is not optional.** `build_config` trains with FSDP
  `param_dtype=bfloat16`; TAU's student partitions include pre-Ampere cards
  (V100, RTX 2080, Titan Xp, Quadro RTX 8000) with no bf16 support. `env.sh`
  therefore defaults `UNTAUGHT_CONSTRAINT` to
  `geforce_rtx_3090|a5000|a6000|a100|l40s`. Don't clear it blindly.
- Path order is deterministic: `NumpyDatasetConfig.glob` sorts its matches
  (`numpy_dataset.py:1902`), so chunk ids are stable across runs given the same
  file set — and `verify_chunk_alignment.py` confirms the deployed file set
  matches the deployed index.
- First run may spend time building dataloader caches if the dataset
  fingerprint differs from the shipped `dataset-348b68...` directory. That is
  automatic, one-time, and does not affect chunk ids (which depend only on the
  sorted file list and the fixed `dataset-common` bucketing files).
- W&B is disabled by default (`WANDB_MODE=disabled`); `build_config` hard-codes
  `WandBCallback(enabled=True)`, which stalls on a node without `WANDB_API_KEY`.
- The two runs write to different `save_folder` roots. They must — `build_config`
  derives its folder name from hyperparameters alone, so with `save_overwrite`
  on, identical settings would silently clobber each other.
- SLURM knobs are named `UNTAUGHT_*` (not `SLURM_*`) deliberately: SLURM owns
  the `SLURM_*` namespace, and launching from inside an `srun --pty bash`
  session would otherwise inherit stale values from the allocation.
