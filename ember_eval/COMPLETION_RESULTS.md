# Completion evaluation of the 2-epoch pair — first run

Run 2026-08-23, SLURM jobs 776326 / 776327 / 776345, `killable`, one RTX 3090
each. Three models, four Pornography splits each, scored by
`completion_eval/evaluate_completion.py`. Records under
`results/completion/`; the two comparison dumps are
`twins2e_completion_comparison.txt` and `released_vs_ourcontrol_comparison.txt`.

## Headline

**No concept-specific difference between the twins is detectable at this sample
size, and the difference that exists is no larger than the one between two
models that were never ablated at all.**

## The instrument works

This is the first thing to establish, and it holds. Concept questions against
neighbouring-domain questions, accuracy, chance 25%:

| model | concept QA | neighbouring domain | spread |
|---|---|---|---|
| released `LMEnt-1B-2E` @ step219344 | 59% | 32% | +27 |
| ours, control 2E | 55% | 34% | +21 |
| ours, ablated 2E | 48% | 30% | +18 |

Every model is far above chance on the concept and near chance on the
neighbouring domain. The declarative stems give these base models something
they can actually answer — the same models score ~0 under EMBER's
letter-parsing protocol (`EVALUATION.md`). Instrument resolution is not the
problem here.

Our control matches the released model within noise (concept QA +0.0400,
p = 0.571), which is `COMPARABILITY.md`'s conclusion reproduced on a better
instrument: **the training recipe is not the limitation.**

## The twins

Drift-free number, `QA − SimdomQA` — concept questions minus neighbouring
domain, which is what isolates an effect specific to the ablated concept.
Negative means the ablation hurt.

| metric | twins (control vs ablated) | p | released vs our control (**no ablation**) | p |
|---|---|---|---|---|
| `p_correct` | +0.0367 | 0.438 | +0.0392 | 0.457 |
| `logp` | +0.4622 | 0.132 | +0.3794 | 0.357 |
| `margin` | −0.0223 | 0.962 | +0.2121 | 0.722 |
| `correct` | −0.0300 | 0.781 | +0.0600 | 0.499 |

**The right-hand column is the finding.** The released model and our control
differ in batch size, optimizer trajectory, code version and random seed — in
everything *except* an ablation. Their concept-specific difference is the same
size as, or larger than, the one between our twins. Whatever separates the
twins on this instrument is not distinguishable from what separates two models
that differ for ordinary reasons.

Note also that two of the four twin numbers carry the **wrong sign**: positive
means the ablated twin did relatively *better* on the concept it never trained on.

## The noise floor, shown directly

The ablated twin scores **42% on `QA_train` and 54% on `QA_test`** — a 12-point
swing between two random halves of the same question set, from one model in one
job. The control's halves sit at 56% / 54% and the released model's at
60% / 58%.

n = 50 per split. A 12-point swing around an effect worth ~4 points is the
entire difficulty, made visible in a single number.

## What is deliberately not claimed

The only per-subset result under p = 0.05 is `logp` on **SimdomQA**: −0.4894,
p = 0.036 — the specificity control, which should be flat.

The tempting reading is "the control moved most, so it is all noise." That
exact claim, at p = 0.002, is **already recorded as retracted** in
`EVALUATION.md` (correction 2), where it turned out to be an `acc_raw`
artifact. A similar pattern under a different metric is a reason for caution,
not a licence to repeat the retracted conclusion.

What can be said without reaching: the pooled `margin` difference is −0.4694
(p = 0.044) and is **the same size on both subsets** (−0.4805 concept, −0.4582
neighbouring). A uniform shift across a concept and an unrelated domain is what
run-to-run drift looks like, not what concept removal looks like.

## Caveats

- Both comparisons are n = 100. Two underpowered measurements agreeing that
  each is near zero establishes that neither is detectable here — not that the
  effect is absent.
- `mean confidence` runs 0.75–0.84. Below the 0.95 saturation warning in
  `evaluate_completion.py`, but high enough that `soft_accuracy` is compressed.
- Metric choice is not load-bearing: all four families agree, and the records
  keep `logp_conditional` / `logp_null` per option, so `acc_per_char` and the
  rest are recomputable without a GPU.
- The 2-epoch control finished ~9.7% of its steps on an H200 (`PROVENANCE.md`).
  Irrelevant to these behavioural numbers; it matters for weight-space work.

## What this changes

The behavioural instrument has now been improved about as far as it can be —
better prompts, four metric families, a paired permutation test, and an external
reference model — and it still shows nothing concept-specific. Two possibilities
remain that no question-answering evaluation can separate:

1. the ablation left no measurable trace, or
2. it left one that 200 questions cannot resolve.

**Perplexity on the held-out chunks separates these directly** — 2,546 chunks,
ids in each run's `untaught_blacklist.json` under `entities[0].chunk_ids`,
millions of tokens, no prompt and no metric choice. `EVALUATION.md` ranked it
second; after this run it is first, and it is still unrun.
