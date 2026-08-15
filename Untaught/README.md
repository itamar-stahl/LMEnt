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
├── submit_full_170m_twins.sh  submit the 170M pair (both jobs)
├── submit_full_1b_twins.sh    submit the 1B pair (both jobs)
├── configs/
│   ├── train_170m_control.yaml       200-step smoke pair
│   ├── train_170m_no_harry_potter.yaml
│   ├── train_170m_control_full.yaml  170M one-epoch pair, studentkillable
│   ├── train_170m_no_harry_potter_full.yaml
│   ├── train_1b_control_full.yaml    1B one-epoch pair, 4x a100
│   └── train_1b_no_harry_potter_full.yaml
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

Once per person, on the TAU cluster (`ssh user@slurm-client.cs.tau.ac.il`).
Every member of the group does exactly the same thing — there is nothing to
edit and no per-user configuration:

```sh
# 1. your own Anaconda, under your own directory -- never in $HOME (quota).
#    Answer the prefix prompt with .../$(whoami)/anaconda3, and conda init: yes.
cd /home/morg/NLP_2526b/$(whoami)
wget repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
conda config --add pkgs_dirs /home/morg/NLP_2526b/$(whoami)/anaconda3/pkgs

# 2. your own clone (no submodules -- OLMo-core is committed into the repo)
git clone <github-url> /home/morg/NLP_2526b/$(whoami)/LMEnt

# 3. your own env, from the file in the repo
conda env create -f /home/morg/NLP_2526b/$(whoami)/LMEnt/environment.yml

# 4. from now on, this is the only command
cd /home/morg/NLP_2526b/$(whoami)/LMEnt/Untaught
. ./activate_env.sh
```

**Why it needs no configuration.** [`framework/env.sh`](framework/env.sh) keeps
two roots apart:

| | resolves to | holds |
|---|---|---|
| `LMENT_USER_ROOT` | `/home/morg/NLP_2526b/$(whoami)` | your clone, your conda, your run folders |
| `LMENT_SHARED_ROOT` | `/home/morg/NLP_2526b/stahli` | the dataset, the index, Elasticsearch |

`whoami` answers the same on the login node and on a compute node, so the same
checkout works for everyone: code and outputs are yours, the ~1 TB of data is
read from one shared copy that nobody duplicates. Every variable is an
override-able default, so `export LMENT_SHARED_ROOT=...` before sourcing if
your data lives somewhere else.

**Elasticsearch is one service for the group.** It runs on the login node as
stahli's process and everyone reaches it on `localhost:9200`; you need no read
access to the install itself. `activate_env.sh` starts it only if `ES_HOME` is
writable by you — otherwise it tells you to ask its owner, rather than failing
halfway through a start.

**Regenerating `environment.yml`** (only when dependencies change):

```sh
conda env export > $LMENT_ROOT/environment.yml
```

Step-by-step version for a new group member: [SETUP.md](SETUP.md).

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

A full training pair, both jobs at once:

```sh
./submit_full_170m_twins.sh     # 170M, 1 titan_xp each, studentkillable
./submit_full_1b_twins.sh       # 1B, 4 a100 nodes each, research partition
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
`./submit_full_170m_twins.sh` again. Each submission gets a fresh run folder, and the trainer looks
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

`job.constraint` asks SLURM for GPU features (`|` means OR). The published
feature list covers the whole cluster, but `studentkillable` holds only two of
them — that is the entire menu here:

| feature | nodes | GPUs | capability | fp32 | memory | compile | native bf16 |
|---|---|---|---|---|---|---|---|
| `titan_xp` | s-002, s-003, s-006 | 23 | 6.1 | 12.1 TFLOPS | 12 GiB | no | no |
| `geforce_rtx_2080` | s-004, s-005 | 16 | 7.5 | 10.1 TFLOPS | 8 GiB | yes | no |

The full configs request **`titan_xp`**. Neither card has native bf16 — that
needs capability 8.0 — so bf16 is emulated on the fp32 pipeline, and there the
TITAN Xp's higher fp32 throughput, 12 GiB and wider memory bus beat the 2080;
the 2080's tensor cores never engage, because they accelerate fp16, not fp32.
The 2080's one advantage is `torch.compile` (7.5 against 6.1), worth roughly
what the throughput gap costs, but 8 GiB is tight at this microbatch and the
TITAN Xp is the card the smoke runs already proved. Asking for a single feature
also puts both twins on identical silicon.

**One epoch takes ~10 days per model**, measured (~4K tokens/s, 3.6B tokens,
~109K steps) — not estimated. Both twins run in parallel, so that is also the
wall clock, spread over ~11 submissions of the 1-day maximum. An a100 would do
it in ~9 hours; there is no a100 in this partition. If ~10 days is too long, the
lever is `train.max_duration_*`: switching to `value: 30000, unit: "steps"`
gives a ~3-day pair that is still a properly controlled comparison, just of a
less-trained model. Both configs must change together.

**Does the card change the final model?** Yes, slightly, and not in a way that
favours either twin. Nothing here changes the *math*: same data order, same step
count, same global batch (gradient accumulation absorbs any microbatch change),
same seed. What differs is floating-point *rounding* — kernel choice and
reduction order differ per architecture, so two runs on different cards diverge
after enough steps and never end bitwise identical. That divergence behaves like
noise, on the order of a re-run with a different seed, and it is far smaller than
the effect the ablation is measuring. Pinning both twins to `titan_xp` removes
even that: same architecture, same kernels, same rounding. Every run also records
what it actually got in `run_environment.json`, so the pair can be checked after
the fact rather than assumed — worth doing, since a widened constraint or a
config edit could silently split them across card types.

### The 1B pair

`configs/train_1b_*_full.yaml` is the same experiment at `olmo2_1B` (d_model
2048, 18 layers, ~1.3B parameters) on a research-group partition:

| | 170M pair | 1B pair |
|---|---|---|
| partition | `studentkillable` | `gpu-<research-group>` — **replace this** |
| GPUs | 1 × `titan_xp` | 4 × `h100` on one node |
| memory | 64 GB | 128 GB |
| global batch | 32,768 tokens | 131,072 tokens |
| rank microbatch | 2,048 | 16,384 |
| steps per epoch | ~109K | ~27.5K |
| peak lr / warmup | 5e-4 / 1000 | 4e-4 / 2000 |
| `torch.compile` | off (capability 6.1) | **on** (9.0) |
| bf16 | emulated | **native** |
| one epoch | ~10 days | **~10–15 h** |

The configs carry no comments by request; the reasoning is here instead.

**`partition` is a placeholder.** `gpu-<research-group>` is not a real partition
— `sbatch` will reject it until you put your group's name there, in both files.

**One node, four GPUs — not four nodes.** The cluster's batch style is the one
on the SLURM page: ask for resources with `#SBATCH`, then just run the program.
The batch script runs on a single host, so `torchrun` starts ranks on that host
only; spreading over 4 nodes would need a rendezvous, which is not how this
cluster is documented to work (`srun` there is for interactive testing, and it
cannot take a script with arguments). Four A100s on one node is also the faster
arrangement — FSDP shards over NVLink instead of the network.

**Why microbatch 16,384.** Every H100 is 80 GB — unlike the A100, there is no
40 GB variant to hedge against, so both twins get the same value whichever node
they land on. At 16,384 tokens a rank holds roughly 5 GB of sharded model and
optimizer state, ~21 GB of activations and ~10 GB of logits: about 36 GB of 80,
with room for fragmentation. 131,072 ÷ 4 ranks ÷ 16,384 is exactly 2
micro-batches per step. Doubling again would need ~67 GB and is not worth the
risk.

**Timing.** ~10–15 h for one epoch, so a run should finish inside a single
1-day job. That is 2.9e19 FLOPs (6ND at 1.34B parameters and 3.6B tokens)
against 4 H100s at 35–40% of peak bf16 — SXM at the fast end, PCIe at the slow
one. Both twins run in parallel.

**Checkpoints are much bigger here** — roughly 16 GB each (1.3B params plus
AdamW moments in fp32) against ~2 GB for the 170M. At
`checkpoint_save_interval: 5000` that is ~6 per run, ~100 GB per run, ~200 GB
for the pair. Check your quota before submitting.

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
