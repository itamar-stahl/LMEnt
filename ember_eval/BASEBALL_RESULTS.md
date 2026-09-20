# The Baseball arm: ablation, EMBER erasure, and what the two disagree about

Second concept after Ancient Rome, run 2026-09-10/11. Same control, same recipe,
same instruments — deliberately, because every axis held fixed is one that
cannot explain a difference between the two concepts.

**Headline, in one line:** the ablation is the strongest this project has
measured (+0.4933 nats/token, 2.1x Rome's), and the erasure **undershoots it on
text while producing a clear question-level effect**. The excluded-model
question contrast is not distinguishable from between-twin variation, so a
question-level overshoot ratio is not established (see the correction below).

## The three models

| | what it is |
|---|---|
| control twin | job 853707, step 54832. **Shared with the Ancient Rome arm** — both ablations mask an already-composed batch, so one control serves every arm. |
| ablated twin | job 873471, step 54832. 43 QIDs / 80,466 chunks (0.7663% of corpus) held out of the loss. Zero guard leaks across five training windows. |
| EMBER-erased | the control with feature 10 of cell rank100 / g_sparsity 0.02 / seed 44 erased at **delta 10.0**. 72 embedding rows changed, `lm_head` untouched. Job 877959. |

---

# 1. The ablation

## Held-out chunk loss — the strongest result in the project

n = 3,000 held-out and 5,004 control chunks, paired per chunk id, seed 42, fp32.
All 3,000 scored chunks verified to be inside the twin's own held-out set.

| | Baseball | Ancient Rome |
|---|---|---|
| **diff-of-diffs** | **+0.4933 nats/token** | +0.2349 |
| dz | **+2.070** | +1.052 |
| control-set drift | +0.0058 (dz 0.091) | +0.0035 (dz 0.085) |
| chunks made worse | **100.0%** | 99.8% |
| median / p90 / max | +0.497 / +0.789 / +2.19 | +0.172 / +0.413 / +2.04 |
| chunks beyond 3 nats | 0.0% | 0.0% |

2.1x Rome's effect at double the effect size, control set flat. The distribution
is uniform — median at the mean, nothing past 3 nats — the never-learned-it
signature, and the opposite of the shattered tail `ERASURE_RESULTS.md` records
for Rome's erasure.

The control finds baseball chunks **easier** than average text (1.78 vs 2.30):
sports prose is formulaic. The ablated twin finds them exactly average (2.28 vs
2.30). The ablation removed the model's home-field advantage on baseball.

**General capability intact.** OLMES `sciq`, n = 1,000: `acc_per_char` 0.7170
control vs 0.6970 ablated — 1.40 SE, within noise.

## The question instruments rank the two concepts the other way

| instrument | n | Baseball vs Rome |
|---|---|---|
| held-out chunk loss | 3,000 | Baseball **2.1x stronger** |
| continuous MC (`gold_per_char`) | 50/cell | Baseball **weaker**: -0.146/-0.089 vs -0.326/-0.288 |
| MC accuracy | 50/cell | Baseball **weaker**: -0.060/-0.040 vs -0.160/-0.160 |
| cross-concept null (`pmi_per_char`) | 50/cell | Baseball **not established**: rank 3/9, z -1.02 |

Not merely noisier — **they invert the ordering of two concepts.**

### Why accuracy in particular sees nothing

Baseball QA train, per question: 22 both-correct, 7 control-only, 4
ablated-only, 17 both-wrong — net **-3 questions**. But the gold answer scored
lower under the ablated model on **42 of 50 questions (84%)**, and on the 22
questions *both* answered correctly the gold score still fell **-0.1245
(p = 0.00000)**.

The ablated model is consistently and significantly less sure about baseball. A
4-option argmax records that only when gold drops below a distractor — 7 times.
The 4 questions the ablated model *gained* are noise, and that churn is roughly
three times the signal.

### Part of the gap is question alignment, measured

For each question, retrieve the corpus chunks best supporting it and ask what
fraction the ablation removed, normalised by each concept's base rate:

| concept | held out | base rate | mean locality | **enrichment** | >50% held out |
|---|---|---|---|---|---|
| Baseball | 80,466 | 0.767% | 0.575 | **75x** | 67/100 |
| Ancient Rome | 65,844 | 0.628% | 0.714 | **114x** | 82/100 |

Rome's questions are **1.5x better aligned** with what its ablation removed.
Rome is asked for facts that live in Ancient Rome articles ("in what year was
Rome founded"); Baseball is asked how the game works ("what object does the
pitcher throw toward the batter"), and rules knowledge is restated corpus-wide.

**Alignment differs by 1.5x while the QA effect differs by 2.2-3.2x, so this
explains part of the gap and not all of it.** The likely remainder is text
predictability: formulaic team and season prose is learned very well and costs
a great deal of likelihood when removed, without carrying many of the facts the
questions ask. That is a hypothesis, not a measurement.

Caveat: locality is an Elasticsearch retrieval proxy for where an answer is
supported, not ground truth about where the model learned it.

---

# 2. The erasure

Feature 10, cell rank100 / g_sparsity 0.02 / seed 44 — chosen on feature quality
before any erasure ran (`lment-ember-grid-bb/GRID_NOTES.md`). The **same cell
Rome's erasure used**, arrived at independently; feature 10 is the structural
analogue of Rome's feature 11.

| | Baseball feature 10 | Rome feature 11 |
|---|---|---|
| ratio_abs | **14.4388** | 6.4643 |
| unique maximum in cell | yes, 2nd = 5.1266 | yes, 2nd = 4.9001 |
| num_concept / num_neutral | 72 / 12 | 76 / 19 |
| judge confidence | 0.99 | 0.99 |

## The delta search bracketed an optimum, unlike Rome's

| delta | qa_retention | specificity | objective |
|---|---|---|---|
| 2.0 | 0.810 | 1.000 | 0.320 |
| 5.0 | 0.333 | 1.000 | 0.800 |
| **10.0** | **0.238** | **1.000** | **0.865** <- chosen |
| 50.0 / 100.0 / 200.0 | 0.238 | 1.000 | 0.865 |

The objective saturates at delta 10; the tie-break `(objective, -delta)` takes
the smallest delta on the plateau. **Rome's objective rose monotonically to 200,
the last value in its grid** — it stopped for lack of candidates, which
`ERASURE_RESULTS.md` identifies as the mechanism of its overshoot. Baseball's
edit is 20x gentler and sits at a genuine optimum.

**The structural flaw is unchanged: specificity is 1.000 at every delta.** The
column carries no information here either, so the objective is efficacy alone.
Baseball escapes the overshoot only because its efficacy saturates early — not
because the objective became better behaved. Do not read this as the metric
having been fixed.

Integrity: `selected_feature_ids: [10]`, `k_features_embed: 1`,
`n_tokens_edited: 72`, `integrity.passed: true`, only
`model.embed_tokens.weight` changed. The materialised model was independently
verified: exactly 72 embedding rows differ from the control and they are exactly
`edited_token_ids`; the other 199 tensors are bit-identical; the untied
`lm_head.weight` is unchanged.

---

# 3. Does the erasure reach never-having-learned?

The two instruments initially appear to disagree about the sign of the
residual. The later cross-concept control shows that the question-side
excluded-model denominator is not established, so only the text residual
supports a directional comparison to never-training.

## On text: it UNDERSHOOTS

| contrast | Baseball | Ancient Rome |
|---|---|---|
| ablated - control (what never-training did) | **+0.4933** | +0.2349 |
| erased - control (what the erasure did) | **+0.3368** | +1.3461 |
| **erased - ablated (the residual)** | **-0.1564** | **+1.1111** |
| ratio erased/ablated | **0.68x** | **5.73x** |

Baseball's erasure reaches **68% of never-training**, and the residual is
negative — the erased model is still measurably better at baseball text than the
twin that never saw it. **Rome's overshot by 5.73x. Same method, same control,
same pipeline, opposite direction.**

Collateral damage is also gone: control-set drift +0.0073, and against the
ablated twin +0.0014 at **p = 0.072, not significant** — on general text the
erased model and the never-trained twin are indistinguishable. Rome's erasure
moved the control set +0.0332 at p = 0.0000.

## On questions: the erasure effect is clear, but the exclusion target is not

`pmi_per_char` on gold, paired per question.

| tier | contrast | train | test |
|---|---|---|---|
| **Baseball QA** | ablated - control | -0.0835 (p=0.033) | -0.0452 (p=0.35) |
| | erased - control | **-0.2535** (p=1.9e-07) | **-0.1935** (p=1.2e-05) |
| | **residual** | **-0.1700** (p=0.0014) | **-0.1482** (p=0.024) |
| **Simdom (other sports)** | ablated - control | +0.2395 (p=0.0038) | +0.2283 (p=0.0042) |
| | erased - control | +0.0261 (p=0.42) | +0.0025 (p=0.96) |
| | residual | -0.2133 (p=0.0066) | -0.2258 (p=0.0013) |

On questions the erasure has a clear negative effect on both splits. The raw
excluded-minus-control contrast is much smaller, but it is not distinguishable
from the between-twin floor; dividing by it to report a multiplier or calling
the difference a verified overshoot is therefore not defensible.

Note the Simdom row: the **ablated twin is significantly *better* than the
control on other sports** (+0.2395, +0.2283). The later direct control reproduced
most of this shift with the no-Rome twin, showing that it is a
between-run/question-set offset rather than a Baseball specificity benefit.

## Reconciling the two

Not a contradiction so much as the day's lesson again. The erasure edits 72
embedding rows for baseball *tokens*, which hits token-level question scoring
hard; the ablation removed 80,466 chunks of baseball *text*, which the chunk
loss sees and the questions largely do not. The two interventions are not
commensurable, and which one looks "bigger" depends entirely on whether you
measure tokens or text.

## Weight space: the two edits are unrelated

| | value |
|---|---|
| `rel_edit_size` | 0.0250 — the erasure update norm is 2.5% of the base-model parameter norm |
| `cosine` | **0.0010** |
| `progress_along_target` | 2.64e-05 |
| `residual` | 1.0003 |
| `model.embed_tokens.weight` alone | rel_edit 0.1003, cosine 0.0043 |

**Cosine 0.001 — essentially orthogonal.** The erasure makes no measurable
progress toward the never-trained model. Whatever it achieves, it does not
approach the twin, and this is the cleanest statement of the three.

---

# 4. Grid findings worth keeping

**Baseball's embedding signature is far cleaner than Rome's.** All 27 cells
carry at least one judge-accepted feature (45 accepted, 43 eligible). Rome's
single-cell attempt had all 24 eligible features rejected; even after its full
grid only 9 of 27 cells carried one. Pornography never cleared the prefilter.

**Stock EMBER sentences are not the confound `grid/README.md` feared** — here:

| sentence source | cells with >=1 | accepted | eligible | degenerate |
|---|---|---|---|---|
| stock EMBER | 27/27 | 45 | 43 | 2 |
| harvested from the held-out chunks | 26/27 | 44 | 37 | **7** |

Both find a judged feature in essentially every cell. The one systematic
difference runs *against* corpus-faithful sentences: harvesting from the
concept's own chunks makes them more concept-concentrated, so more features end
up with zero neutral support.

**A guard the pipeline lacks.** `ratio_abs` is undefined when `num_neutral == 0`
— the denominator is an epsilon — and such features reach absurd values (up to
298,701) while looking ideal to the judge, since every one of their tokens is
on-concept. The judge accepted both of the stock arm's degenerate features, at
confidence 0.99 and 0.95. They were excluded under a `num_neutral > 0` rule
written down **before** any judging. `stats_embed.csv` should arguably carry a
validity flag.

---

# 5. Scope and caveats

**One concept, one pair, one seed, one cell, one delta.** Everything in
`ROME_RESULTS.md`'s scope section applies unchanged.

**The delta was not re-picked after seeing any twin comparison.** It came from
EMBER's own sweep objective, as Rome's did. The cell was chosen on feature
quality before the erasure ran. Both rules were recorded in advance, in
`GRID_NOTES.md`, precisely because `ERASURE_RESULTS.md` documents that choosing
an erasure hyperparameter by the size of the effect it produces is what
retracted two `acc_raw` claims here.

**Input embeddings only.** EMBER edited 72 rows of `model.embed_tokens.weight`;
the untied `lm_head.weight` was untouched, while the ablation shaped the whole
network throughout training. The two interventions are not the same kind of
object — the cosine of 0.001 is that fact in numeric form.

**Three hardcoded-label bugs were found and fixed while producing this.**
`cross_concept_null.py` printed a literal "Ancient Rome" beside a correctly
retargeted value; `erasure_vs_twins.py` raised `KeyError` on a `LABEL` dict
built at import time, then printed every Baseball section under the heading
"Rome". In each case the numbers were right and the label was wrong — which is
the dangerous form. Rome's outputs still reproduce exactly after the fixes
(-0.3745 / -0.2797, diff-of-diffs +0.2349). Anyone quoting these scripts on a
non-Rome concept before 2026-09-11 should re-check what they read.

## Artifacts

| path | what |
|---|---|
| `/home/dcor/galbarak2/hf-models/lment-1b-nobaseball-2e-b131k` | the ablated twin (mirrored to `/home/morg` backups; private Hub repo `GalBarak/lment-1b-nobaseball-2e`) |
| `/home/dcor/galbarak2/hf-models/lment-1b-baseball-erased-b131k` | the erased model |
| `/home/dcor/galbarak2/lment-baseball-check/results/` | chunk-loss JSONs, three-way contrasts, weights, locality |
| `/home/dcor/galbarak2/lment-ember-grid-bb/` | stock feature grid, judge results, `GRID_NOTES.md` |
| `/home/dcor/galbarak2/lment-ember-grid-bb-corpus/` | corpus-faithful grid arm |
| `.../Ember-on-LMEnt/runs/Baseball_lment-1b-control-2e-b131k_20260911_140756/` | the erasure run, `report.json` |

Jobs: conversion 876641; completion eval 876691-876699; chunk loss 876631 /
876701 / 878766; OLMES 876774; embedding health 877472; feature grids 876632
(stock) / 876805 (corpus); judge 876772 / 877306 / 877307 (stock), 877437-877439
(corpus); erasure 877959; erased-model evals 878765 / 878766 / 878772.

---

# CORRECTION (2026-09-11): the QA arm's denominator is not established

Section 3's line "on questions the erasure removes 3-4x what the ablation did"
**should not be read as a ratio.** It divides -0.2535 by -0.0835, and the
denominator is `z = -1.02`. Dividing by a quantity indistinguishable from zero
does not give 3x; it gives an undefined number. The defensible statement is
narrower: **the Baseball erasure has a clear question-level effect and the
Baseball ablation does not, so the two cannot be placed in a ratio.**

## What the cross-concept null already said, read directly

`results/cross_concept_null_baseball.json`, `pmi_per_char` on gold:

| split | observed | null mean (n=8) | null sd | z | rank |
|---|---|---|---|---|---|
| QA train | -0.0835 | -0.0185 | 0.0638 | **-1.02** | 3/9 |
| QA test | -0.0452 | -0.0292 | 0.0543 | **-0.30** | 2/9 |
| Simdom train | +0.2395 | -0.0270 | 0.0511 | **+5.21** | **1/9** |
| Simdom test | +0.2283 | -0.0295 | 0.0538 | **+4.79** | **1/9** |

## The direct demonstration, which needs no null at all

Score a twin that **never ablated baseball** (`norome2e`) on the same Baseball
questions:

| Baseball questions | own ablation | Rome-untaught | concept-specific remainder |
|---|---|---|---|
| QA train | -0.0835 | **-0.0638** | -0.0197 |
| QA test | -0.0452 | **-0.0355** | -0.0097 |
| Simdom train | +0.2395 | **+0.1920** | +0.0475 |
| Simdom test | +0.2283 | **+0.2083** | +0.0200 |

**76% of the apparent QA effect is reproduced by a model that saw every baseball
chunk.** Ancient Rome for contrast: -0.3745 own against **+0.0116** for the
unrelated twin -- essentially all of Rome's effect is the concept.

## Section 2's unexplained Simdom result is now explained

That section records the ablated twin scoring significantly *better* than the
control on other sports (+0.2395, +0.2283) and says "nothing in the design
predicts that, and it is not explained here." It is now: **Rome-untaught does the
same thing** (+0.1920, +0.2083) on the same questions, and Rome's ablation
removed no baseball. Both twins are outliers on the Baseball-Simdom set while the
null over eight *other* Simdom sets sits at -0.027. The anomaly belongs to that
question set, not to either ablation, and it is not a knowledge effect.

## The mechanism, and why chunk loss is immune

There is **one shared control** (job 853707) behind every arm. Any run-level
idiosyncrasy of that single run -- it trained on ~0.7% more data than either
twin, with its own preemption history -- enters every `ablated - control`
contrast identically. The QA instrument has no defence against it: it compares
two models on one question set and calls the difference the ablation.

The held-out chunk loss does have a defence, by construction. Its diff-of-diffs
subtracts each model's own loss on **length-matched control chunks**, which
absorbs exactly this offset -- the point `HELDOUT_RESULTS.md` makes as "each twin
is its own baseline". That is why Baseball's chunk-loss result (+0.4933, dz 2.07,
100% of chunks worse) stands while its QA result does not, and it is an argument
for treating chunk loss as **the** instrument rather than one of two.

**Nothing here touches Section 1's chunk-loss findings or the erasure's own
`erased - control` contrasts** (the erased model is control-derived and carries
no twin offset). What it removes is the ablation-side denominator on questions,
and with it the "3-4x" reading.
