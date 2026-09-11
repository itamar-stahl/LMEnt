# LMEnt-1B twin, TWO EPOCHS: ABLATED — Baseball core + teams held out

The ablated half of the **Baseball** arm of the 2-epoch twin pair, at batch
131,072. Trained to the full 54,832 steps with 43 baseball QIDs masked from the
loss.

**Its control already exists and is shared with the Ancient Rome arm.** Both
ablations mask an already-composed batch, so both twins traverse identical
batches and one control serves every arm:

    /home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k        <- the control (job 853707)
    /home/dcor/galbarak2/hf-models/lment-1b-nobaseball-2e-b131k     <- this model
    /home/dcor/galbarak2/lment-rome-check/hf/norome-2e-step54832    <- the Ancient Rome ablated twin

## What was removed

| | |
|---|---|
| blacklist | `Untaught/blacklists/baseball_core_teams.json` — CORE + TEAMS, 43 QIDs |
| chunks excluded | **80,466** — 0.7663% of the 10,491,928-chunk corpus |
| thresholds | hyperlinks 1.0, entity-linking 0.6, coref 0.6, coref-cluster 0.6 |
| primary entity | `Q1163715` (Baseball), 38,401 chunks |

For scale: this is **1.22x the Ancient Rome footprint** (65,844 chunks, 0.628%)
and **32x** the retired Pornography subject (2,546 chunks, 0.0242%).

Chunks were masked from the **loss**, not removed from the data: they still
occupy batch slots, so batch composition, data order and step count are
identical to the control's. The twins differ by exactly the masked gradient
contributions.

## Proof the ablation fired, across all five training windows

This run took five SLURM windows. Every one loaded the blacklist and every one
reports **zero guard leaks** — no blacklisted chunk ever reached the loss
through the all-masked-batch guard.

| window | last step | instance-slots excluded | guard leaks |
|---|---|---|---|
| `20260905_153110` | 6,282 | 19,330 | 0 |
| `20260906_101708` | 43,224 | 74,076 | 0 |
| `20260908_120920` | 50,016 | 11,596 | 0 |
| `20260909_182701` | 54,178 | 19,096 | 0 |
| `20260910_083914` | **54,832** | 7,042 | 0 |

**Do not add that column up and call it a path total.** It sums to 131,140, but
the windows overlap: window 4 reached step 54,178 and then died writing its
step54000 checkpoint (`OSError: [Errno 5]`, `filesystem.py:105`), leaving a
12-of-16-shard partial, so window 5 resumed from the last *valid* checkpoint at
step 52,000 and re-trained 52,000-54,178. Those steps are counted twice. The
figure is a lower bound on work done, not a count along the realised path.

As with the Pornography card: the excluded counter **resets per window**, so the
framework's closing line reports only the last one (`7047`), and it prints with
a thousands separator, so a grep for `[0-9.]+` truncates `1,074` to `1`.

## Training

| | |
|---|---|
| duration | **2 epochs** = 54,832 steps |
| final CE loss | 2.473 → **perplexity 11.86** |
| final job | `873471`, `COMPLETED 0:0`, 3h24m, n-h200 |
| hardware | H100 for windows 1-4, **H200 for window 5** — see below |

Everything else matches the b131k recipe: `olmo2_1B` (18 layers, d_model 2048,
16 heads), 3.6B-token LMEnt Wikipedia corpus, global batch 131,072 tokens, rank
microbatch 16,384, AdamW peak LR 4e-4, warmup 2,000, cosine to 4e-5, VSL
`grow_p2` over 8 cycles, seed 12536.

**Hardware deviation, stated because it matters for weight-space work.** The
final 2,832 steps ran on an H200 rather than an H100, after n-102's GPU1 failed
NVML enumeration twice in one day while SLURM still reported the node healthy
(`nvmlDeviceGetHandleByIndex(1) failed`, jobs 871261 and 872110), and n-102 held
the only free H100s in the partition. The control has a comparable deviation —
its last ~9.7% of steps ran on an H200. Measured cross-GPU drift on this project
is 3e-5 and flips nothing in evaluation, but it is a real asymmetry and belongs
next to any weight-space claim. Config: `train_1b_no_baseball_core_teams_2e_b131k_h200.yaml`,
which changes **only** `partition` — `job.name` and `constraint` are deliberately
untouched, because neither is in `RESUME_IGNORED_FIELDS` and changing either
would have silently restarted training from random init.

## Provenance

| | |
|---|---|
| SLURM job | `873471` (final window) |
| Run folder | `untaught-no-baseball-core-teams-1b-2e-b131k-h100_20260910_083914` |
| Step | `54832` / 54,832 (2 epochs, complete) |
| Converted from | `.../checkpoints/olmo2_1B_0.0004_131072_0.05_2/step54832` |
| Converted by | `OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py`, job `876641` |
| Tokenizer | taken from the control's own export, not the Hub |
| Converted on | 2026-09-11 |

Verified after conversion: 16 `.distcp` shards plus `.metadata` at the source,
201 tensors across 2 safetensors shards at the destination with a complete
index, and `config.json` matching the control (18 layers, d_model 2048, vocab
100,352, `tie_word_embeddings: false`).

**The tokenizer came from the control export rather than `allenai/dolma2-tokenizer`**
because the cached OAuth token had expired and the Hub returned 401. The two are
equivalent and this is strictly better for twin comparability: vocabulary
identical (100,278 entries), added tokens identical, and merges identical in
content — they differ only in serialisation (`['Ġ','Ġ']` versus `'Ġ Ġ'`), a
transformers version format change.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "/home/dcor/galbarak2/hf-models/lment-1b-nobaseball-2e-b131k"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, torch_dtype="auto")
```

## Backups

Mirrored across independent filers (`/home/dcor` is netapp1, `/home/morg` is
netapp2), every leg verified with `rsync -c` content hashing:

- HF export → `/home/morg/NLP_2526b/galbarak2/backups/lment-2e/hf-models/lment-1b-nobaseball-2e-b131k`
- distcp checkpoint → `/home/dcor/galbarak2/backups/lment-2e/checkpoints/untaught-no-baseball-core-teams-1b-2e-b131k-h100_20260910_083914/step54832`

The HF copy is **not** a checkpoint backup: the `.distcp` directory is
`model_and_optim`, fp32 master weights plus optimizer moments at 15 GB. The HF
safetensors are a lossy weights-only derivative at 5.1 GB. distcp → HF works;
HF → distcp does not.

> Keep this file and the copy inside the model directory in sync — nothing does
> it automatically.
