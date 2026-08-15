# Group setup

Everyone runs the same steps. Nothing to edit, no per-user config.

**Work on `c-003`.** Elasticsearch listens on `localhost:9200` there, and the
client half of the framework needs it. Other hosts will fail at submit time.

```sh
ssh <user>@c-003
```

## 1. Anaconda (once)

Per the cluster docs (*User packages → Example using Anaconda*):

```sh
cd /home/morg/NLP_2526b/$(whoami)
wget repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
```

The installer is interactive. Two answers matter:

| prompt | answer |
|---|---|
| install prefix | `/home/morg/NLP_2526b/<you>/anaconda3` — **never your home**, the quota is too small |
| `conda init?` | `yes` |

Re-login (or `. ~/.bashrc`) so `conda` is on your PATH, then move the package
cache off your home, or `conda env create` will fill it:

```sh
conda config --add pkgs_dirs /home/morg/NLP_2526b/$(whoami)/anaconda3/pkgs
```

A newer `Anaconda3-*-Linux-x86_64.sh` from the same archive works too, and
solves faster — `environment.yml` pins python 3.12.

## 2. Clone (once)

```sh
git clone <github-url> /home/morg/NLP_2526b/$(whoami)/LMEnt
```

No `--recurse-submodules`: OLMo-core is committed into the repo.

## 3. Environment (once)

```sh
conda env create -f /home/morg/NLP_2526b/$(whoami)/LMEnt/environment.yml
conda activate lment
```

Creates env `lment` with everything pinned. ~10 min. If `conda` isn't found,
you skipped `conda init` — run
`. /home/morg/NLP_2526b/$(whoami)/anaconda3/etc/profile.d/conda.sh` first.

## 4. Every session

```sh
cd /home/morg/NLP_2526b/$(whoami)/LMEnt/Untaught
. ./activate_env.sh
```

Source it, don't execute it — it has to change *your* shell. It sets every
path, activates `lment`, and checks Elasticsearch.

## 5. Verify

```sh
python tests/run_local_tests.py
python -m framework.node.train_untaught configs/train_170m_control.yaml --check
```

Three suites, then a config that builds. Both take seconds. If they pass,
you're done.

## 6. Submit

```sh
. ./framework/client/sub_builder.sh configs/train_170m_control.yaml   # one run
./submit_full_170m_twins.sh                                           # the pair
```

Each submission creates `runs/<job>_<date>_<time>/` in **your** checkout with
the config, the blacklist artifact, `job.slurm`, logs and checkpoints. Track
with `squeue --me`.

---

## What's yours, what's shared

| | where | |
|---|---|---|
| clone, conda, runs, checkpoints | `/home/morg/NLP_2526b/$(whoami)/` | yours, writable |
| dataset, index, Elasticsearch | `/home/morg/NLP_2526b/stahli/` | shared, read-only |

`framework/env.sh` derives the first from `whoami` and hard-codes the second.
Same on the login node and on compute nodes, so nothing needs configuring.

To override anything, export it **before** sourcing — every variable is a
`:=` default:

```sh
export UNTAUGHT_RUNS_DIR=/somewhere/else
. ./activate_env.sh
```

## Elasticsearch

One server for the group, owned by stahli, running on `c-003`. You reach it at
`localhost:9200`; you don't need read access to the install.

`activate_env.sh` prints `Elasticsearch is up` when it's fine. If it says it is
not yours to start — it's down, **ask stahli**. Don't start your own: it would
fail on the shared data directory.

Only the *client* half needs it (turning blacklist QIDs into chunk ids, before
submission). GPU nodes never touch it — the resolved chunk ids travel in the
run folder.

## When it breaks

| Symptom | Fix |
|---|---|
| `cannot cd to .../LMEnt/Untaught` | clone is missing or in the wrong place — step 2 |
| `conda.sh missing; falling back to PATH` | miniconda not at `$(whoami)/anaconda3` — step 1 |
| `Elasticsearch is down` | ask stahli; don't start one |
| `blacklist artifact not found` at pre_train | you ran `sbatch` directly; always go through `sub_builder.sh` |
| `Permission denied` under `stahli/LMEnt` | expected — that clone is private, use your own |
| job vanishes from `squeue` | `studentkillable` preempted it; resubmit, it resumes from the last checkpoint |

Full detail: [README.md](README.md). Cluster smoke test: [SMOKE_TEST.md](SMOKE_TEST.md).
