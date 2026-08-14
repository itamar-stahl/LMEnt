# Smoke testing Untaught on the cluster

> **Just want it all checked?** One command does everything below, plus a real
> submission, and writes a report you can send back:
>
> ```sh
> cd /home/morg/NLP_2526b/stahli/LMEnt/Untaught
> sh tests/remote/run_remote_tests.sh
> ```
>
> It sets up the environment itself. When it finishes, send
> `runs/remote_test_<date>_<time>/report.log`. Options: `--quick` (skip the
> slow dataset/alignment phases), `--no-submit` (checks only, no SLURM job),
> `--wait N` (minutes to wait for the job, default 20).
>
> The rest of this document is the manual walkthrough of the same ground.

Two 200-step runs — a control and a "Harry Potter"-ablated twin — that prove
the whole pipeline before any real compute is spent. Both are submitted the
same way; the only difference between them is the config.

Everything below runs on the login node (`ssh <user>@slurm-client.cs.tau.ac.il`).

## 0. Environment (once per shell)

```sh
cd /home/morg/NLP_2526b/stahli/LMEnt/Untaught
. ./activate_env.sh
```

That single source leaves this shell with the `lment` conda env active, all
paths and ES variables set, and Elasticsearch running (it starts it if down —
a cold start takes ~30–60 s and is logged to `runs/elasticsearch.log`).
Nothing is read from `~/.bashrc`, so this works identically for any user.

## 1. Preflight (GPU-free, do these in order)

```sh
# a. all three local suites — seconds, no cluster resources
python tests/run_local_tests.py

# b. the configs build and every path resolves — seconds
python -m framework.node.train_untaught configs/train_170m_control.yaml --check
python -m framework.node.train_untaught configs/train_170m_no_harry_potter.yaml --check

# c. THE decisive check: proves ES chunk_id == dataset instance index on this
#    deployment, by comparing decoded chunk text against indexed text
#    (a few minutes; run once per deployment)
python tests/verify_chunk_alignment.py --config configs/train_170m_control.yaml -n 25
```

**If (c) fails, do not train** — the blacklist would exclude the wrong chunks.
It means the deployed index and dataset disagree (stale snapshot, renamed or
missing tokenized part).

## 2. Inspect the ablation before spending queue time

```sh
# which QIDs does the corpus actually use for the name?
python -m framework.client.es_blacklist resolve --name "Harry Potter"

# how many chunks would this config hold out, with samples?
python -m framework.client.es_blacklist count \
    --config configs/train_170m_no_harry_potter.yaml --preview 5
```

`count` uses the same entities, thresholds and index the run will use, so its
number is the run's number. A count of 0 means wrong QIDs — fix
`blacklists/harry_potter.json` before submitting.

## 3. Submit — one command per run

```sh
. ./framework/client/sub_builder.sh configs/train_170m_control.yaml
. ./framework/client/sub_builder.sh configs/train_170m_no_harry_potter.yaml
```

Each submission prints (and records in `client.log`) its run folder:

```
runs/<job.name>_<date>_<time>/
├── config.yaml               the exact config this run trains on
├── untaught_blacklist.json   the exact exclusion (explicitly empty for control)
├── job.slurm                 what went to sbatch — pure strings, no env vars
├── run_wrapper.sh            what the GPU node executed (literal paths)
├── client.log                this submission's log
├── log.out / log.err         the job's output (appear when it starts)
└── checkpoints/              parameters by step (step100, step200 for smoke)
```

Track with `squeue --me`; watch with `tail -f <run_dir>/log.out`.
Expect a long quiet stretch before step 1 on a cold dataset cache — the first
`dataset.prepare()` scans the corpus once; later runs reuse the cache.

## 4. What success looks like

Both jobs reach step 200 with loss falling from ~10–11. Then compare the logs:

| Signal in log.out | Control | Ablated |
|---|---|---|
| header `blacklist` line | `(none) -- CONTROL run` | the blacklist path |
| `[untaught] loaded N chunk ids` | absent | N > 0 |
| `train/untaught excluded instances` | absent | non-zero on some steps |
| `train/masked instances` (upstream's) | absent | matches the above |
| `train/untaught guard leaks` | absent | **must stay absent** |
| total steps | 200 | 200 — identical, that is the point |

And the artifacts:

```sh
# control: explicitly empty, and says so
cat runs/untaught-control-170m_*/untaught_blacklist.json | head -3

# ablated: per-entity chunk counts, index, thresholds
python - <<'PY'
import json, glob
art = json.load(open(sorted(glob.glob("runs/untaught-no-hp-170m_*/untaught_blacklist.json"))[-1]))
print(art["num_chunks"], "chunks;", [(e["qid"], e["num_chunks"]) for e in art["entities"]])
PY
```

## 5. Common failures

| Symptom | Meaning | Fix |
|---|---|---|
| `blacklist artifact not found` at pre_train | run folder wasn't built by sub_builder.sh | submit via sub_builder.sh, never bare sbatch |
| ablated run shows all-zero exclusions | QIDs matched nothing | `resolve` the name, fix `blacklists/*.json` |
| CUDA OOM | small/busy GPU | `adapt_to_gpu` already shrinks the microbatch; check `nvidia-smi` in log.out for a card another job filled |
| pages of `WON'T CONVERT` warnings | pre-Ampere GPU, compile disabled automatically | harmless for smoke; prefer a newer card for real runs |
| job vanishes from `squeue` | `studentkillable` preempted it | resubmit; checkpoints resume from `checkpoints/` |

Every question of "what did this run actually do?" is answered inside its run
folder — that is the design.
