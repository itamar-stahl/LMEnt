# LMEnt-1B twin, TWO EPOCHS: ABLATED — Artificial Intelligence held out

The ablated half of the **Artificial Intelligence** arm of the 2-epoch twin
pair, at batch 131,072. Trained to the full 54,832 steps with 31 AI-related
QIDs masked from the loss.

**Its control is shared with the Ancient Rome and Baseball arms.** All three
ablations mask an already-composed batch, so every twin traverses identical
batches and one control serves every arm:

    /home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k     <- the control (job 853707)
    /home/dcor/galbarak2/hf-models/lment-1b-noai-2e-b131k        <- this model
    /home/dcor/galbarak2/hf-models/lment-1b-norome-2e-b131k      <- the Ancient Rome ablated twin
    /home/dcor/galbarak2/hf-models/lment-1b-nobaseball-2e-b131k  <- the Baseball ablated twin

## What was removed

| | |
|---|---|
| blacklist | `Untaught/blacklists/ai_core.json` — 31 QIDs |
| chunks excluded | **19,818** — 0.189% of the 10,491,928-chunk corpus |
| primary entity | `Q11660` (Artificial intelligence), plus 30 related QIDs |

The per-entity counts sum to 25,361, so 5,543 ids are shared between entities
and the deduplicated union is exactly the declared 19,818.

Chunks were masked from the **loss**, not removed from the data: they still
occupy batch slots, so batch composition, data order and step count are
identical to the control's. The twins differ by exactly the masked gradient
contributions.

## Proof the ablation fired

Every leg's startup log confirms `loaded 19818 chunk ids`.

Verifying the exclusion *count* is harder here than for Rome, because the
`train/untaught excluded cumulative` counter **resets per leg** and the legs
overlap across restarts. The final leg is the one clean window: it excluded
**6,300** instance-slots over steps 46000–54832, 16.11% of the run, against
`2 x 19,818 = 6,384` predicted for that window — **98.7% of prediction**.

**Do not compare this to Rome's 99.95% and conclude the ablation was leakier.**
Rome ran in a single window, so its count is a whole-run figure; this is one
window of eight. They are different measurements.

### The blacklist is 68% precise, and that is measured, not assumed

Audit job `884973`, on 400 sampled blacklisted chunks:

    PRECISION: 272/400 blacklisted chunks are unambiguously AI = 68.0%
    LEAKAGE:   11/6000 random chunks are AI = 0.18%
    TRUE-POSITIVE coverage = 70.1%

Coverage (recall) is 70.1%; precision is 68.0% — two different numbers that are
easy to conflate. About a third of the held-out set is entity-annotation
sweep-in: ordinary text that merely mentions a related entity.

This does not invalidate the arm. The ablated twin received no gradient from
those chunks whether or not each is topically AI, so "did withholding these
chunks leave a trace" is answered correctly either way. What it does is
**dilute** the measured trace, making every effect size a floor rather than an
inflated figure. Rome's and Pornography's precision was never audited, so do
**not** rank the three ablations by these numbers.

## Training

| | |
|---|---|
| duration | **2 epochs** = 54,832 steps |
| final CE loss | 2.457 → **perplexity 11.67** |
| final job | `905819`, `COMPLETED`, ended 2026-09-18T03:12 |
| hardware | **mixed across eight legs** — see below |

Everything else matches the b131k recipe and is identical to the control's:
`olmo2_1B` (18 layers, d_model 2048, 16 heads), 3.6B-token LMEnt Wikipedia
corpus, global batch 131,072, rank microbatch 16,384, AdamW peak LR 4e-4,
warmup 2,000, cosine to 4e-5, VSL `grow_p2` over 8 cycles, seed 12536, one GPU.

### The run took eight legs on three different cards

**State this next to any weight-space claim.** The control (`853707`) ran on
`n-h200` throughout; this twin did not:

| leg | partition | node | card | elapsed | exit |
|---|---|---|---|---|---|
| `888131` | `gpu-h200` | `n-h200` | H200 | 8:51:38 | FAILED |
| `893211` | `gpu-h200` | `n-h200` | H200 | 20:48:56 | CANCELLED |
| `896181` | `gpu-h200` | `n-h200` | H200 | 5:06:47 | FAILED |
| `897181` | `gpu-h100-killable` | `n-102` | H100 | 0:06:35 | FAILED |
| `897192` | `killable` | `n-601` | A6000 | 11:52:57 | CANCELLED |
| `898135` | `gpu-h200` | `n-h200` | H200 | 2:40:06 | CANCELLED |
| `898528` | `gpu-h200` | `n-h200` | H200 | 3:03:28 | FAILED |
| **`905819`** | `gpu-h200` | `n-h200` | H200 | 7:19:21 | **COMPLETED** |

Most of the run is H200, matching the control. The `n-102` leg died in 6m35s and
contributed nothing — that node's GPU 0 is a physically dead card. The `n-601`
leg is the real deviation: ~12 hours of training on an A6000. Measured
cross-GPU drift on this project is 3e-5 and flips nothing in evaluation, but it
is a genuine asymmetry against a control that never left `n-h200`.

Two legs died the same way — **`OSError: [Errno 5] Input/output error`
while writing a checkpoint**, at step 46000 and again at step 48000.

That is not disk-full (quota exhaustion is errno 122, and the volume had
headroom both times). `/home/dcor` is mounted **`soft`** on every node
(`vers=3, rsize=32768, wsize=32768, soft, timeo=600, retrans=2`), so the kernel
returns EIO rather than blocking once a burst crosses the timeout. A 15 GB async
checkpoint save fires 16 writer threads at a 32 KB write size — roughly 490,000
write RPCs in a burst, itself enough to cross it. Both deaths show the same
signature: steady 2.1 s/step, step time ballooning to 90–170 s exactly when the
save starts, EIO two to three minutes later. No other job on the filer hit EIO
in those windows, so it is self-inflicted congestion, not an outage.

`checkpoint_preflight.sh` in the run folder contains the damage — it refuses to
resume from a checkpoint with fewer than 16 shards and moves it aside — but the
underlying fragility is **unfixed**. A retry-with-backoff around `_write_items`
in `OLMo-core/.../checkpoint/filesystem.py` would turn a dead job into a
30-second pause, and is the recommended fix before the next twin.

## Is this model readable at all

Yes. Embedding health (job `906839`): init std 0.019732, final row-norm 1.4369,
predicted-from-decay 0.4894, **ratio 2.936**, row-cosine median 0.2917 —
matching the Rome b131k twins (2.9374 / 2.9361) to within 0.01%. That agreement
is the cleanest evidence these runs differ only in the ablation. Against a ratio
of 1.00 for the destroyed pre-fix pair, whose embeddings sat at noise.

## What it showed

Full write-up: `ember_eval/AI_RESULTS.md`. In brief — the target-concept QA drop
is real and survives prior correction (`logp_null` unchanged, p = 0.24 / 0.50),
the held-out chunk-loss trace is +0.1561 nats/token at dz **1.83** (the most
*consistent* of the three subjects), and the apparent +12-point SimdomQA *gain*
is a prior artifact that collapses to −0.014 once `pmi/char` cancels the prior.

## Provenance

| | |
|---|---|
| SLURM job | `905819` (final leg) |
| Run folder | `untaught-no-ai-core-1b-2e-b131k-h100_20260912_162646` |
| Step | `54832` / 54,832 (2 epochs, complete) |
| Converted from | `.../checkpoints/olmo2_1B_0.0004_131072_0.05_2/step54832` |
| Converted by | `convert_checkpoint_to_hf.py`, job `906830`, 2026-09-18 |
| Tokenizer | taken from the control's own export, not the Hub |

**The tokenizer came from the control export rather than `allenai/dolma2-tokenizer`**
because the cached OAuth token had expired and the Hub returns 401. The two are
equivalent, and this is strictly better for twin comparability.

## A caveat that cannot be undone

The intermediate checkpoints `step5000`–`step54000` (11 x 15 GB, ~165 GB) were
deleted on 2026-09-18 to clear filer quota. `step0` and `step54832` are kept and
verified, and **nothing published depends on the intermediates** — every number
comes from step54832, or step0 for embedding health. But a **step-matched**
comparison against the control is now foreclosed for AI without retraining, and
retraining means eight legs across six days. See the `CHECKPOINTS_REMOVED.md`
left beside the run folder.

## Backups — this model has none, as of 2026-09-19

Unlike its control and the Rome and Baseball twins, **neither artifact of this
twin is mirrored to a second filer**:

- the HF export is only at `/home/dcor/galbarak2/hf-models/lment-1b-noai-2e-b131k`
- the distcp checkpoint is only at
  `LMEnt-baseball-alltopics/Untaught/runs/untaught-no-ai-core-1b-2e-b131k-h100_20260912_162646/checkpoints/.../step54832`
  — also on `/home/dcor`

Both are on netapp1. A single filer holds every copy of a model whose
intermediate checkpoints were already deleted, so losing it means re-running an
eight-leg, six-day training. Mirroring costs ~20 GB: the HF export to
`/home/morg/.../backups/lment-2e/hf-models/`, the checkpoint to a
`checkpoints/` directory under the same backup root.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "/home/dcor/galbarak2/hf-models/lment-1b-noai-2e-b131k"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, torch_dtype="auto")
```

These are **base** models with no instruction tuning. Letter-parsing evaluators
score them near zero — given a multiple-choice prompt they do not emit a letter
at all. Score option text by log-likelihood, prefer a declarative stem over
`Question: ...\nAnswer:`, and normalise per character. See
`ember_eval/EVALUATION.md`.

> Keep this file and the copy in `Untaught/model_cards/` in sync — nothing does
> it automatically.
