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
├── activate_env.sh          LOGIN NODE: source it — cd + vars + conda + Elasticsearch
├── framework/
│   ├── env.sh               shared variables — the only place paths are named
│   ├── conda.sh             shared conda activation
│   ├── client/              needs Elasticsearch; never imported by a training job
│   │   ├── sub_builder.sh   THE submit flow: builds a run folder, sbatch job.slurm
│   │   ├── es_blacklist.py  entity QIDs -> chunk ids, via Elasticsearch
│   │   └── prepare.py       fills the run folder: config copy, artifact, job.slurm,
│   │                        run_wrapper.sh
│   └── node/                no Elasticsearch dependency at all
│       ├── set_node_env.sh  COMPUTE NODE: cd + vars + conda (sourced by run_wrapper.sh)
│       ├── train_untaught.py  wraps examples/kas/train.py, attaches the callback
│       ├── exclusion.py     ChunkExclusionCallback — the ~15 lines that matter
│       ├── run_folder.py    the run-folder contract: layout + artifact reading
│       └── config_env.py    ${VAR} expansion for configs; reads env.sh on miss
├── run.sh                   submit the full training pair (both jobs)
├── configs/
│   ├── train_170m_control.yaml       200-step smoke pair
│   ├── train_170m_no_harry_potter.yaml
│   ├── train_170m_control_full.yaml  one-epoch training pair
│   └── train_170m_no_harry_potter_full.yaml
├── blacklists/
│   └── harry_potter.json    QIDs to hold out; named by the config above
├── runs/                    one self-contained folder per submission (see below)
├── tests/
│   ├── run_local_tests.py   runs all three local suites, one summary
│   ├── test_local_units.py  component behaviour: callback, thresholds, run folder
│   ├── test_local_integration.py  every OLMo-core contract, against the real code
│   ├── test_local_refactoring.py  package completeness & structural integrity
│   ├── remote/              REMOTE: one script, end-to-end, writes a report
│   ├── archive/             pre-refactor suites, kept for reference only
│   └── verify_chunk_alignment.py  REMOTE preflight: proves chunk_id alignment empirically
└── SMOKE_TEST.md            step-by-step guide for the two smoke runs on the cluster
```

Upstream code is **not modified**. `train_untaught.py` imports `build_config`
from `examples/kas/train.py` verbatim, so the model, optimizer, dataset, VSL
curriculum and data order stay exactly upstream.

---

## Setup

On the TAU cluster (`ssh user@slurm-client.cs.tau.ac.il`):

```sh
cd $LMENT_ROOT/Untaught
. ./activate_env.sh               # that is the whole setup
```

Sourcing `activate_env.sh` leaves **this** shell with the `lment` conda env
active, every path/ES variable set, and Elasticsearch running (it starts it if
it is down). It reads nothing from `~/.bashrc`, so it behaves the same for any
user. Adjust `framework/env.sh` only if your paths differ from
`/home/morg/NLP_2526b/stahli`.

Both entry points `cd` to `$LMENT_ROOT/Untaught` first, so the working directory
and `UNTAUGHT_ROOT` are the same fixed path on every node. Batch jobs get the
same environment from `framework/node/set_node_env.sh`, sourced by each run
folder's `run_wrapper.sh` — minus Elasticsearch, which the GPU nodes cannot
reach.

Preflight, in order, all GPU-free:

```sh
# 1. all three local suites (seconds)
python tests/run_local_tests.py

# 2. config builds, paths resolve, blacklist readable (seconds)
python -m framework.node.train_untaught configs/train_170m_control.yaml --check

# 3. THE decisive check — proves on this deployment that ES chunk_id
#    equals the dataset instance index, by comparing decoded chunk text
#    against the indexed text for random chunks (a few minutes; run once)
python tests/verify_chunk_alignment.py --config configs/train_170m_control.yaml -n 25
```

If step 3 fails, **do not train** — the blacklist would exclude the wrong
chunks. It means the deployed index and dataset directory disagree (stale
snapshot, missing/renamed tokenized file).

---

## Submitting a run

The full training pair, both jobs at once:

```sh
./run.sh
```

Or any single config:

```sh
. ./framework/client/sub_builder.sh configs/train_170m_control.yaml
. ./framework/client/sub_builder.sh configs/train_170m_no_harry_potter.yaml
```

Everything about the job — its name, SLURM resources, hyperparameters, and the
ablation — is in the config; `sub_builder.sh` holds only the shared flow. Each
submission creates its own **run folder**, `runs/<job.name>_<date>_<time>/`,
which is both the working record and the save location:

```
runs/untaught-no-hp-170m_20260814_153000/
├── config.yaml               verbatim copy — later edits to configs/ can't touch this run
├── untaught_blacklist.json   the exact exclusion (explicitly empty for a control run)
├── job.slurm                 what was submitted — pure strings, zero env variables
├── run_wrapper.sh            what the GPU node executed
├── client.log                the submission-side log
├── log.out / log.err         the job's stdout / stderr
├── run_environment.json      the GPU it actually got, and what that changed
└── checkpoints/              saved parameters, by step
```

`sub_builder.sh` sets up the client environment, fills the folder via
`framework.client.prepare`, validates every file exists (and that `job.slurm`
contains no unresolved variables), then submits with `sbatch job.slurm`.

Both generated scripts are **literal strings, no shell variables** — reading
either one tells you exactly what ran, with nothing to look up. The node just
executes `run_wrapper.sh`, whose launch line names the trainer and the run
folder and nothing else:

```sh
torchrun --standalone --nproc-per-node=1   /…/framework/node/train_untaught.py untaught-control-170m_20260814_212936
```

The config is not named because it is implied: the trainer knows where run
folders live (`framework/node/run_folder.py`) and that each holds its own
`config.yaml`, artifact and `checkpoints/`. Tracing any run back is: open its
folder.

### Long runs on a preemptible partition

`studentkillable` caps a job at **1 day** and can preempt it at any time, while
one epoch is ~109K steps — so a full run takes several submissions. Just run
`./run.sh` again. Each submission gets a fresh run folder, and the trainer looks
one folder back: if the newest previous run of the same job left a checkpoint,
it continues from it; otherwise it starts from a random init. The run header
says which:

```
  resuming from    : …/untaught-control-170m-full_20260814_212936/checkpoints/…
  resuming from    : (nothing found) -- random init
```

A candidate qualifies only if its folder name starts with this run's
`job.name` **and** the `config.yaml` it actually trained on matches this one
field for field — the whole `untaught` block included. Only the scheduling and
logging knobs may differ, since those do not change the experiment:

```
job:    partition, max_time_minutes, nodes, ntasks, cpu_mem_mb,
        cpus_per_task, gpus, resume_from_previous_run
train:  checkpoint_save_interval, checkpoint_ephemeral_save_interval,
        checkpoint_save_async, checkpoint_save_overwrite,
        metrics_collect_interval, cancel_check_interval,
        wandb_cancel_check_interval, eval_tasks, eval_interval
```

So a changed seed, curriculum, dataset path or blacklist starts a clean run
instead of silently continuing a different experiment, and the rejected
candidate is logged with the field that disqualified it. (Matching on
upstream's checkpoint *folder name* would not do: it encodes only
model/lr/batch/wd/duration-**value**, so it cannot even tell one epoch from one
step, let alone a different seed or a different blacklist.)

`job.constraint` is compared too, with one exception: an **empty** constraint
means "any card", so it is compatible with anything, and only two runs that each
demanded a card and demanded *different* ones are treated as different
experiments.

Permanent checkpoints land every 10,000 steps; an ephemeral one every 500 steps
caps what a preemption can cost.

### The GPU lottery

`job.constraint` asks SLURM for GPU features (`|` means OR). The full configs
request **`a100`** and nothing else: it is the cluster's fastest card, and
pinning both twins to one feature means they train on identical silicon — same
kernels, same precision, same speed — so nothing about the hardware can be
mistaken for an effect of the ablation. The cost is queue time; a100 nodes are
the ones everyone wants.

To schedule sooner, widen it to `a100|l40s|a6000|a5000|geforce_rtx_3090` — every
card on the cluster at CUDA capability ≥ 8.0, so the run still gets native bf16
*and* `torch.compile` whichever one it lands on, at the price of the twins
possibly landing on different models. Below that line the cluster's other cards
(`tesla_v100`, `quadro_rtx_8000`, `geforce_rtx_2080`, `titan_xp`) have no native
bf16, and `titan_xp` loses compile as well: measured at ~4K tokens/s, it turns
one epoch into ~10 days instead of ~half a day.

Estimates for one epoch (3.6B tokens, ~109K steps), assuming ~30–35% of peak
bf16 — typical for a 170M model at sequence length 2048:

| feature | one epoch | notes |
|---|---|---|
| `a100` | ~9–12 h | **what the full configs request** — may fit one 1-day job |
| `l40s` | ~19 h | |
| `a6000` | ~22 h | |
| `a5000` | ~31 h | ~2 submissions |
| `geforce_rtx_3090` | ~2 days | GeForce halves tensor throughput at fp32 accumulate |
| `titan_xp` | ~10 days | **measured**, not estimated: no bf16, no compile |

**Does the card change the final model?** Yes, slightly, and not in a way that
favours either twin. Nothing here changes the *math*: same data order, same step
count, same global batch (gradient accumulation absorbs any microbatch change),
same seed. What differs is floating-point *rounding* — kernel choice and
reduction order differ per architecture, so two runs on different cards diverge
after enough steps and never end bitwise identical. That divergence behaves like
noise, on the order of a re-run with a different seed, and it is far smaller than
the effect the ablation is measuring. Pinning both twins to `a100` removes even
that: same architecture, same kernels, same rounding. Every run also records what
it actually got in `run_environment.json`, so the pair can be checked after the
fact rather than assumed — worth doing, since a widened constraint or a config
edit could silently split them across card types.

### Inspecting the ablation before submitting

```sh
python -m framework.client.es_blacklist resolve --name "Harry Potter"   # which QIDs exist?
python -m framework.client.es_blacklist count --config configs/train_170m_no_harry_potter.yaml --preview 5
```

Which entities are held out is one file — `blacklists/harry_potter.json`, a list
of QIDs with comments — and the training config names it:

```yaml
untaught:
  blacklist: "blacklists/harry_potter.json"
```

The path is relative to `Untaught/`. To hold out something else, write another
file in `blacklists/` and point a config at it.

**Elasticsearch is only reachable from the login node**, so the QIDs are turned
into chunk ids there, before submission, and the result travels to the GPU node
as a file in the run folder:

```
login node                                   GPU node
──────────                                   ────────
blacklists/harry_potter.json  (QIDs)
config: thresholds, case_sensitive
      │  framework.client.prepare   (run by sub_builder.sh)
      ▼
<run_dir>/untaught_blacklist.json ─────────► pre_train: load into {chunk_id: qid}
                                                     │
                                             pre_step: dict lookup per row → mask
```

`framework.client.prepare` runs one ES query per entity and writes a readable JSON
recording the ids, the index used, the thresholds and per-entity counts — so a
run and the exact exclusion it was trained with stay together. The training job
never talks to Elasticsearch; a missing artifact is a hard error, not a silent
control run.

**A run folder without an artifact fails at `pre_train`** with an error naming
`sub_builder.sh` — an ablated run can never silently train as a control.

**Verify the QIDs before trusting a run.** `resolve` reports which QIDs the
corpus actually attaches to that name, with mention counts; `count` reports how
many chunks the file would remove. The defaults are `Q8337` (the series) and
`Q3244512` (the character); a franchise, its characters and its individual books
are all separate Wikidata entities, so decide how wide your concept is.

### What to look for

| Signal | Control | Ablated |
|---|---|---|
| `blacklist` line in the header | `(none) -- CONTROL run` | path + the QIDs |
| `train/untaught excluded instances` | absent | non-zero on some steps |
| `train/masked instances` (OLMo-core's own) | absent | matches the above |
| `train/untaught guard leaks` | absent | **must stay absent** (see caveats) |
| total steps | 200 | 200 (identical — that is the point) |

If the ablated run shows all zeros, the blacklist did not match anything —
check your QIDs with the `resolve` command above. The full remote walkthrough
is in [SMOKE_TEST.md](SMOKE_TEST.md).

---

## Going from smoke to a real run

In both configs, change:

```yaml
max_duration_value: 1
max_duration_unit: "epochs"
```

One epoch is ~109K steps (3.6B tokens ÷ 32,768 tokens/step). For a 170M model
that is roughly 4–8 hours on one H100-class GPU. Then also:

- `disable_downstream_eval: false` and set `eval_interval` to `1000`
- raise `checkpoint_save_interval` to `1000` (the paper's cadence, 110
  checkpoints/epoch)
- bump `job.max_time_minutes`; `studentkillable` caps at 1 day, so a full epoch
  needs several resumes or a longer partition

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
- **The GPU you get is not the GPU you want.** `build_config` hard-codes
  `compile=True` and FSDP `param_dtype=bfloat16`; `studentkillable` hands out
  pre-Ampere cards (Titan Xp, V100, RTX 2080) where triton cannot compile at
  all and bf16 is emulated. `--constraint` was tried and rejected by this
  cluster, so `train_untaught.adapt_to_gpu` handles it at runtime instead:
  compile off below capability 7.0, and `rank_microbatch_size` scaled down to
  fit the card (2048 on a 12GB Titan Xp, unchanged at 8192 on ≥48GB). Gradient
  accumulation keeps the 32,768-token global batch, so the optimizer math and
  the control/ablated comparison are unaffected.
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
