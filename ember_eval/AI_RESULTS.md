# The Artificial Intelligence twins: the ablation, and whether erasure reaches it

Two 1B models on the same two Wikipedia epochs, identical except that one held
every chunk mentioning **Artificial Intelligence** out of the loss. Finished
2026-09-18. This is the third subject after Ancient Rome (`ROME_RESULTS.md`) and
Pornography (`HELDOUT_RESULTS.md`), and the first where the never-learned twin
and the post-hoc erasures are all scored on one scale.

**Headline, in one line each:**

- The ablation is **real, concept-specific and prior-robust**, but far louder on
  the text than on the questions — `dz` **1.83** on withheld chunks against a
  QA drop whose McNemar test does not reach significance.
- The neighbouring-domain "gain" the twin appears to show is a **prior
  artifact** and vanishes under `pmi_per_char`.
- **EMBER's erasure lands on never-having-learned**, which Rome's did not.
  Neither MLP method does: SNMF moves nothing, RMU damages everything.

---

## The models

| | Control | Ablated (no-ai) |
|---|---|---|
| Job (final leg) | `853707` | `905819` |
| Run folder | `untaught-control-1b-2e-b131k_20260905_221156` | `untaught-no-ai-core-1b-2e-b131k-h100_20260912_162646` |
| Held out | nothing | 19,818 chunks, 31 QIDs |
| Steps | 54,832 / 54,832 | 54,832 / 54,832 |
| Final CE loss | 2.473 | **2.457** |
| Final perplexity | 11.85 | **11.67** |
| Exit | `COMPLETED` | `COMPLETED` |
| HF model | `hf-models/lment-1b-control-2e-b131k` | `hf-models/lment-1b-noai-2e-b131k` |

Same recipe as the Rome pair: `olmo2_1B`, lr 4e-4, warmup 2000, global batch
131,072, rank microbatch 16,384, `kas_vsl` `grow_p2` over 8 cycles, seed 12536,
one GPU. The control is shared with the Rome pair, so a twin comparison here is
against the same weights `ROME_RESULTS.md` used.

### The run took eight legs, and the reason matters

Training spanned legs `888131`, `893211`, `896181`, `897181`, `897192`,
`898135`, `898528`, `905819`. Two of them died the same way: **`OSError: [Errno
5] Input/output error` while writing a checkpoint**, at step 46000 and again at
step 48000.

That is not a disk-full error (quota exhaustion is errno 122, and the volume had
headroom both times). `/home/dcor` is mounted **`soft`** on every node:

    vers=3, rsize=32768, wsize=32768, soft, timeo=600, retrans=2

Under a soft mount the kernel gives up after roughly `timeo` x `retrans` and
returns EIO to the application rather than blocking. The 15 GB async checkpoint
save fires 16 writer threads at a 32 KB write size — roughly 490,000 write RPCs
in a burst — and that burst is itself enough to cross the timeout. Both deaths
show the same signature: steady 2.1 s/step, then step time balloons to 90-170 s
exactly when the save starts, then EIO two to three minutes later. No other job
on the same filer in the same window hit EIO, so it is self-inflicted
congestion, not an outage.

`checkpoint_preflight.sh` in the run folder contains the damage — it refuses to
resume from a checkpoint with fewer than 16 shards and moves it aside — but the
underlying fragility is unfixed. **A retry-with-backoff around `_write_items` in
`OLMo-core/.../checkpoint/filesystem.py` would turn a dead job into a 30-second
pause**, and is the recommended fix before the next twin.

## Did the ablation happen

Yes. The blacklist holds **19,818 chunk ids across 31 entities** (Q11660 plus 30
related QIDs); the per-entity counts sum to 25,361, so 5,543 ids are shared
between entities and the deduplicated union is exactly the declared 19,818.

Verifying the exclusion count is harder here than for Rome, because the
`train/untaught excluded cumulative` counter **resets per leg** and the legs
overlap across restarts. The final leg is the one clean window: it excluded
**6,300** instance-slots over steps 46000-54832, which is 16.11% of the run;
2 x 19,818 predicts 6,384 for that window, so the observed figure is **98.7%**
of prediction. Every leg's startup log also confirms `loaded 19818 chunk ids`.

### The blacklist is 68% precise, and that is already measured

Audit job `884973`, on 400 sampled blacklisted chunks:

    PRECISION: 272/400 blacklisted chunks are unambiguously AI = 68.0%
    LEAKAGE:   11/6000 random chunks are AI = 0.18%
    TRUE-POSITIVE coverage = 70.1%

Note these are two different numbers that are easy to conflate: **coverage
(recall) is 70.1%, precision is 68.0%.** About a third of the held-out set is
entity-annotation sweep-in — ordinary text that mentions a related entity.

This does not invalidate anything. The ablated twin received no gradient from
those chunks whether or not each is topically AI, so "did withholding these
chunks leave a trace" is answered correctly either way. What it does is
**dilute** the measured trace, which makes every effect size below a floor
rather than an inflated figure.

## Are these models readable at all

Yes. Embedding health (job `906839`) against the Rome twins:

| | no-ai twin | Rome control | Rome ablated |
|---|---|---|---|
| init std | 0.019732 | 0.019732 | 0.019732 |
| final row-norm | 1.4369 | 1.4377 | 1.4370 |
| predicted from decay alone | 0.4894 | 0.4894 | 0.4894 |
| **ratio** | **2.9357** | 2.9374 | 2.9361 |
| median row cosine to init | 0.2917 | 0.2916 | 0.2916 |

Against 1.00 for the destroyed pre-fix pair. Agreement to within 0.01% across
three independently trained models is the cleanest evidence these runs differ
only in the ablation.

---

# Part 1 — what the ablation did

## Completion eval

50 questions per split, 4 options, chance 25%. All arms scored by
`completion_eval/evaluate_completion.py` against **this checkout's 18-concept
bank**, `gold/char` = `log P(gold | stem) / len(gold)`.

| split | control | ablated | released 2E |
|---|---|---|---|
| QA_train | 44%  −0.9516 | **32%**  −1.1268 | 38% |
| QA_test | 54%  −0.8651 | **48%**  −0.9748 | 56% |
| SimdomQA_train | 54%  −1.0102 | 66%  −0.9067 | 54% |
| SimdomQA_test | 50%  −1.1342 | 52%  −1.0933 | 56% |

Paired per question, ablated minus control:

| split | Δ gold/char | dz | t p | Δ pmi/char | McNemar p |
|---|---|---|---|---|---|
| QA_train | **−0.1752** | −0.56 | **0.0002** | −0.114 | 0.109 |
| QA_test | **−0.1097** | −0.50 | **0.0010** | −0.102 | 0.508 |
| SimdomQA_train | +0.1035 | +0.56 | 0.0003 | −0.014 | 0.031 |
| SimdomQA_test | +0.0410 | +0.26 | 0.0671 | −0.032 | 1.000 |

**The concept drop is real and survives prior correction.** On both AI splits
the gold option's `logp_null` is unchanged (p = 0.24 and 0.50), so this is
missing knowledge rather than a shifted prior, and `pmi/char` moves the same
direction and size.

**The Simdom "improvement" is a prior artifact — do not report it as collateral
benefit.** The ablated twin assigns systematically higher unconditional
probability to those gold options: `logp_null` **+1.68 (p = 0.0001)** on
SimdomQA_train and **+1.18 (p = 0.0009)** on SimdomQA_test. Once `pmi/char`
cancels the prior, the apparent +12-point gain becomes **−0.014**. This is the
same trap that ate Baseball's +22; score `logp_null` alone before believing any
QA-minus-Simdom gap.

Accuracy McNemar is non-significant on both AI splits while the continuous
statistic resolves at p ≤ 0.001 **from identical forward passes** — the n=50
power problem documented in `EVALUATION.md`, not a weak effect.

## Held-out chunk loss — the ablation left a large trace on the text

Jobs `907132` (control) and `907133` (ablated), ~1h25m each. 3,000 held-out
chunks sampled from the 19,818 (`lment-ai-check/ai_blacklist_sample3000.json`,
seed 20260918) plus 5,004 length-matched control chunks, seed 42, fp32.

| set | control 2E | ablated 2E | difference | `dz` | p |
|---|---|---|---|---|---|
| **held-out chunks** (3,000) | 2.6529 | 2.7942 | **+0.1561** | **+1.831** | <0.0001 |
| control chunks (5,004) | 2.3618 | 2.3628 | +0.0017 | +0.050 | 0.0005 |

    DIFFERENCE OF DIFFERENCES: +0.1544 nats/token, label-permutation p < 0.0001

In perplexity: held-out **14.195 → 16.349, +15.2%**; matched unmasked text
**10.610 → 10.621, +0.10%**.

**Read the effect sizes, not the p-values**: `dz` 1.831 against 0.050, a factor
of 37. The control-chunk row is a live example of why — 0.0017 nats reaches
p = 0.0005 at n = 5,004 while being, by any standard that matters, zero.

The control-set agreement (2.3618 vs 2.3628) is a third independent check that
the twins differ only in the ablation, alongside the embedding ratios and the
overall perplexity pair.

### Indexing was proved before the GPU ran

Job `907120`, `check_indexing_concept.py`: **blacklisted 9/12 carry concept
vocabulary, random 0/12**, verdict `INDEXING CONFIRMED`.

9/12 rather than Rome's 12/12 is **expected at 68% precision** (8.2/12
predicted), not an index problem — and the random 0/12 is the half that actually
tests indexing. Of the three misses, one (`Information space analysis`,
"enhanced by **machine intelligence**") is a gap in the marker regex rather than
a bad chunk; the other two carry no AI vocabulary anywhere in their full 2,048
tokens and are the sweep-in the audit predicts.

The marker regex lives in `lment-ai-check/check_ai_indexing.slurm`. It
deliberately omits a bare `\bai\b`: markers compile with `re.I`, so that
alternative would fire on any stray "Ai" and inflate the random-chunk count,
which is the half of the test that can only be lost.

### Against the other two subjects

| subject | held-out diff | `dz` held | control diff | `dz` ctl | diff-of-diffs |
|---|---|---|---|---|---|
| Pornography | +0.1168 | +1.890 | +0.0014 | +0.048 | +0.1154 |
| Ancient Rome | +0.2384 | +1.052 | +0.0035 | +0.085 | +0.2349 |
| **Artificial Intelligence** | +0.1561 | **+1.831** | +0.0017 | +0.050 | +0.1544 |

Rome has the largest mean shift but the *smallest* `dz` — far more per-chunk
variance. AI's trace is the more consistent one, close to Pornography's.

**Do not rank the three ablations on these numbers.** Only AI's blacklist
precision has been audited (68%); Rome's and Pornography's have not, so this
compares differently-clean held-out sets, and AI's dilution makes +0.1544 a
floor.

**Trap:** `lment-rome-check/results/` holds **two** Rome diff-of-diffs files and
neither records which model pair produced it. `rome_heldout_diffdiff.json`
(0.2349, dz 1.052) is the 2E b131k pair — the comparable one;
`rome_heldout_comparison.json` (0.0920, dz 0.647) is the step80000 pair.
Recompute from the raw `ppl_*.json`, which do record model paths. Reaching for
the wrong one understates Rome by 2.5x.

---

# Part 2 — does erasure reach never-having-learned

This is the question the twins exist for. Four post-hoc interventions on the
**control**, all scored by the same harness, same bank, same seed as the twin,
so every number below is mutually comparable.

> **The MLP numbers in `runs/mlp_erasure/` are NOT on this scale.** That harness
> puts the identical control at 44% on AI/QA_test where this one puts it at 54%
> — a 10-point gap on byte-identical weights (verified: both safetensors shards,
> `index.json` md5 and config all match; only a README differs). Comparing
> across the two would be arithmetic across different scales, the same class of
> error as the mid-run `evaluate_completion.py` edit that faked an 18-point
> swing. That is the whole reason every arm here was re-scored.

## The arms

| arm | how it was built | model |
|---|---|---|
| **EMBER** | job `907972`, judge-selected features 26/85/89, `chosen_delta` 5.0 | `hf-models/lment-1b-ai-erased-b131k` |
| **SNMF** | job `908304`, re-applied from cache `snmf_ai_887806` (84 features) | `hf-models/lment-1b-ai-snmf-b131k` |
| **RMU α=100** | job `908316`, L5 hi, steering 846.6, batch 1, 150/150 steps | `runs/mlp_erasure/rmu_ai_L5hi_fixed_908316/model` |
| **RMU α=10** | job `908317`, same, α=10 | `runs/mlp_erasure/rmu_ai_L5hi_fixed_908317/model` |

## Results: each arm minus control, `pmi_per_char`

Negative means the concept was damaged.

| arm | QA_train | QA_test | Simdom_train | Simdom_test |
|---|---|---|---|---|
| **twin** (never learned) | −0.114 | −0.102 | −0.014 *ns* | −0.032 *ns* |
| **EMBER** | **−0.116** | **−0.090** | −0.026 | −0.033 *ns* |
| SNMF | **+0.030** | **+0.049** | +0.056 | +0.062 |
| RMU α=100 | −0.030 | −0.063 | −0.036 | −0.021 *ns* |
| RMU α=10 | −0.145 | **−0.211** | **−0.114** | **−0.131** |

## The residual: distance from the twin

`pmi_per_char`, method minus twin, bootstrap 95% CI on `dz` (20k resamples):

| method | QA_train dz [CI] | QA_test dz [CI] |
|---|---|---|
| **EMBER** | **−0.01 [−0.27, +0.31]** | **+0.04 [−0.23, +0.36]** |
| SNMF | +0.56 [+0.25, +1.00] | +0.56 [+0.27, +0.92] |
| RMU α=100 | +0.33 [+0.05, +0.69] | +0.17 [−0.11, +0.50] |
| RMU α=10 | −0.10 [−0.34, +0.21] | −0.41 [−0.66, −0.16] |

**EMBER is the only method that reproduces the twin's profile.** It matches the
concept damage almost exactly while leaving the neighbouring domain alone, and
its residual straddles zero on both concept splits.

**SNMF moves `pmi` upward on all four splits** (+0.03 to +0.06, all p < 0.001) —
a small systematic improvement, the wrong sign for an erasure. Its residual
excludes zero on the positive side: it falls well short of the twin.

**RMU α=10 is not a null, but it is not erasure either.** It overshoots the
concept (−0.211 against the twin's −0.102 on QA_test, CI excluding zero) while
also damaging the neighbour by −0.114 and −0.131, both significant, where the
twin's collateral is −0.014 and −0.032 and is not. That is roughly **4-8x the
collateral** — general degradation that happens to include the concept. α=100 is
the same thing at lower amplitude.

**Read the equivalence claim carefully.** At n=50 the residual CIs are ±0.3, so
EMBER's result is *"indistinguishable from zero and bounded within ±0.3 dz"*,
**not** proven equivalence. What is solid is the contrast with Rome.

## Rome disagrees, and the mechanism is visible

Rome's EMBER erasure **overshot** its twin: erased−control `pmi/char` −0.652
against twin−control −0.375, residual `dz` **−0.315, CI [−0.598, −0.031],
excluding zero** (`erasure_vs_twins.json`). AI's lands on the twin.

Same method, opposite verdict, and the feature structure differs in a way that
fits. For Rome and Baseball the Gemma judge picked **one dominant feature**,
which was also the unique maximum of `ratio_abs`:

    Rome      feature 11   6.4643  (next 4.9001)
    Baseball  feature 10  14.4388  (next 5.1266)

For AI the judge accepted **three**, and its top pick is only third by ratio:

    feature 94   6.3396   <- judge REJECTED
    feature 55   6.1961   <- judge REJECTED
    feature 89   4.6780   conf 0.99  "artificial intelligence and autonomous machine systems"
    feature 85   3.4576   conf 0.95  "computer vision and machine learning applications"
    feature 26   2.2372   conf 0.85  "data processing, neural network architectures"

A broader concept spread over more, weaker features appears to be one the
embedding erasure removes proportionately rather than overshooting. The erasure
edited **187 embedding rows**.

**Consequence for the config:** no `feature_ratio_threshold` can isolate feature
89, because anything low enough to admit it also admits the two the judge
rejected. So AI cannot use the `mode: threshold` shortcut that
`ember_lment_{rome,baseball}_erase_nojudge_slurm.yaml` use; the judge has to
run, and `constraint: a6000` is load-bearing. See
`ember_lment_ai_erase_slurm.yaml`.

**Judge reproducibility, checked:** the 2026-09-18 run reproduced the original
2026-09-14 accepted set exactly — {26, 85, 89} with bit-identical metric scores,
confirming the cached factorisation was replayed rather than refit. Only feature
26's free-text description differed, and its confidence moved 0.85 → 0.95. Worth
knowing: **0.85 sat exactly at `judge_confidence_threshold`**, so that feature
originally passed by zero margin, and this erasure's composition is one judge
wobble from being two features rather than three.

## Why the RMU null carries weight

Every RMU cell in the Rome depth grid failed `ran_enough_steps` at 47%: the
forget pool is ~283 usable sentences and `n = min(max_num_batches, forget)`, so
batch 4 caps a 150-step request at ~71. **At batch 1 the run reaches 150/150.**

Both AI cells were run that way and are the cleanest in the project:

| | α=100 | α=10 |
|---|---|---|
| cos_forget start → end | 0.0583 → 0.4125 | 0.0617 → **0.5458** |
| `rotation_is_substantial` (≥0.30) | True | True |
| steps | **150/150** | **150/150** |
| retain cosine | 0.9958 | 0.9772 |
| **`failed`** | **[]** | **[]** |

α=10's 0.5458 is the highest rotation measured in this project. So RMU's failure
to erase selectively here is a null **from its best-ever configuration** — a
genuine misdirection that ran its full schedule and preserved the retain set —
not from an under-run job that never really tried.

---

## Where everything lives

| artifact | path |
|---|---|
| Training run | `Untaught/runs/untaught-no-ai-core-1b-2e-b131k-h100_20260912_162646/` |
| Twin model | `hf-models/lment-1b-noai-2e-b131k` |
| Erased models | `hf-models/lment-1b-ai-{erased,snmf}-b131k`, `runs/mlp_erasure/rmu_ai_L5hi_fixed_9083{16,17}/model` |
| Completion results | `LMEnt-ember/ember_eval/results/completion/cmpl_{control2e,noai2e,aierased2e,aisnmf2e,airmu_a100,airmu_a10}_ai_*` |
| Chunk-loss results | `lment-ai-check/results/ppl_{control2e,noai2e}_ai_90713{2,3}.json`, `ai_heldout_diffdiff.json` |
| Embedding health | `lment-ai-check/results/emb_health_noai2e_906839.json` |
| Chunk sample | `lment-ai-check/ai_blacklist_sample3000.json` |
| Feature cache | `lment-ember-grid-ai/features/sp0.02_seed44/` |
| Job scripts | `lment-ai-check/*.slurm` (not under git, matching the other `*-check` dirs) |

### Reclaimed 2026-09-18 (980G → 800G)

All of the following were stripped after the results above were banked. Each
left a `WEIGHTS_REMOVED.md` / `CHECKPOINTS_REMOVED.md` beside it; configs,
tokenizers and every result JSON are intact.

**Cheap to rebuild — minutes:**

- `hf-models/lment-1b-ai-snmf-b131k` (5.1 G) — `snmf_ai_materialize.slurm`
- `runs/mlp_erasure/rmu_ai_L5hi_fixed_9083{16,17}/model` (5.1 G each) —
  `rmu_ai_redo.slurm` with `ALPHA=100` / `ALPHA=10`

**NOT cheap — needs the full retrain, be certain before doing this again:**

- intermediate checkpoints `step5000`–`step54000` (11 x 15 G, ~165 G). `step0`
  and `step54832` are kept and verified. An earlier draft of this file called
  these "cheaply rebuilt", which was wrong: the erased models re-derive in
  minutes, but a checkpoint requires re-running a training that took eight legs
  across six days. Nothing published depends on them — every number here comes
  from step54832, or step0 for embedding health — but a **step-matched**
  comparison against the control is now foreclosed for AI without retraining.

**Do NOT delete** without regenerating first:

- `lment-ember-grid-ai/features/` — the sparse factorisation **does not
  reproduce across GPU models at a fixed seed**, so this cannot be rebuilt, only
  replaced with a different object
- `runs/mlp_erasure/snmf_ai_887806/features.pkl` — same reason
- `step0` and `step54832` of the training run
- the result JSONs, which are the record

Leave a `WEIGHTS_REMOVED.md` beside anything stripped, as the Rome delta-sweep
variants do.
