# Does erasure reach the state of never having trained on the concept?

**No.** On the Ancient Rome pair, EMBER's embedding erasure -- run with its own
delta selection -- **overshoots the concept by 1.7-2.2x and significantly damages
the neighbouring domain, which the ablation did not touch.**

This is the question the twins were built to answer, and `ROME_RESULTS.md` listed
it as the last one still open. It is now measured, for one erasure configuration.

## The three models

| | what it is |
|---|---|
| control twin | job 853707, step 54832. Trained on everything. |
| ablated twin | job 850249, step 54832. 56 QIDs / 65,844 chunks held out of the loss. |
| EMBER-erased | the control, with feature 11 of cell rank 100 / g_sparsity 0.02 / seed 44 erased at delta 200. 76 embedding rows changed, `lm_head` untouched. Job 867391. |

All three scored on the same questions by byte-identical code
(`evaluate_completion.py`, md5 `249cc08c14f35851a945a00bd2f994a2`), with the
rule imported from `metric_bakeoff` rather than reimplemented
(`erasure_vs_twins.py`). Paired per question, `pmi_per_char` on `gold`.

## The result

Mean paired delta, with `dz`:

| contrast | Rome train | Rome test | Simdom train | Simdom test |
|---|---|---|---|---|
| **ablated - control** | -0.3745 (-0.79) | -0.2797 (-0.76) | -0.0027 (-0.01) *ns* | -0.0765 (-0.22) *ns* |
| **erased - control** | **-0.6516** (-0.82) | **-0.6237** (-0.98) | **-0.1894** (-0.42) | **-0.2083** (-0.56) |
| **erased - ablated** | -0.2771 (-0.31) | -0.3439 (-0.54) | -0.1867 (-0.31) | -0.1318 (-0.28) |
| | p=0.031 | p=0.00034 | p=0.0044 | p=0.057 |

The first row reproduces `ROME_RESULTS.md` to the digit (t = -5.60, p = 9.6e-07
and t = -5.34, p = 2.4e-06). It is here as a regression check: if it ever moves,
the scoring changed and nothing else on this page can be trusted.

**Two findings.**

1. **The erasure overshoots.** It removes 1.74x (train) and 2.23x (test) what
   never-training removed, and the residual against the ablated twin is
   significantly negative on both halves. The two interventions do not arrive at
   the same state.

2. **The erasure is not specific; the ablation was.** Next door the ablation is
   flat -- `dz` -0.008 and -0.215, neither significant, and on the validation
   half `ROME_RESULTS.md` found Rome moved *less* there than any of eight
   untouched concepts. The erasure moves it `dz` -0.42 (p = 0.0044) and -0.56
   (p = 0.00024).

## The delta sweep, which is the evidence for the section below

From job 867391's `report.json`. Baseline is the unerased control: Rome QA 0.48,
Simdom 0.38, chance 0.25. `efficacy` is 1 - chance-corrected QA retention;
`specificity` is the same for Simdom, capped at 1.0.

| delta | Rome QA acc | Simdom acc | efficacy | specificity | objective |
|---|---|---|---|---|---|
| 0.5 | 0.48 | 0.38 | 0.000 | 1.00 | 0.000 |
| 1.0 | 0.46 | 0.38 | 0.087 | 1.00 | 0.160 |
| 2.0 | 0.48 | 0.38 | 0.000 | 1.00 | 0.000 |
| 5.0 | 0.40 | 0.44 | 0.348 | 1.00 | 0.516 |
| 10.0 | 0.36 | 0.44 | 0.522 | 1.00 | 0.686 |
| 50.0 | 0.34 | 0.42 | 0.609 | 1.00 | 0.757 |
| 100.0 | 0.34 | 0.42 | 0.609 | 1.00 | 0.757 |
| **200.0** | **0.32** | 0.38 | **0.696** | **1.00** | **0.821** |

Two things to read off it. **Specificity is 1.00 in every row** -- the column
carries no information at all, so the objective is efficacy alone. And **200 is
the last value in the grid with the objective still rising**, so the search never
bracketed an optimum; it stopped because it ran out of candidates.

The efficacy differences past delta 10 are also **one to two questions out of
50** (0.36, 0.34, 0.34, 0.32) against a standard error of about +-0.067, so the
ranking among 10/50/100/200 is noise. Enlarging the grid would pick a different
"winner" with no more meaning. What is not noise is the direction: every step up
in delta costs Rome accuracy, and the sensitive measure shows it costing the
neighbour too.

## EMBER's own specificity metric said 1.00, and that is why it overshot

The erasure run reported `specificity: 1.00` **at every delta in the grid**. Its
specificity is accuracy-based, simdom accuracy went 0.38 -> 0.38, and the metric
is capped at 1.0, so it read as untouched.

This is not merely a metric that missed something. It is **causal**. The delta
objective is efficacy against specificity; with specificity pinned at its cap,
nothing in the objective ever opposed a larger delta, so the search ran to the
top of its grid (200, the endpoint, objective still rising) and kept going past
the point where the neighbour started to suffer. **The blunt metric licensed the
overshoot.**

That is this project's oldest lesson recurring: a 50-question accuracy on base
models that cannot really answer questions is too coarse to steer with. It cost
two retracted `acc_raw` claims (`EVALUATION.md`), and it has now cost an erasure
its specificity. The `pmi_per_char` measure sees what accuracy cannot.

## The accuracy columns, shown rather than asserted (2026-09-09)

The section above says a 50-question accuracy is too coarse to steer with. Here
is the table behind that claim, recomputed from the per-option scores stored in
each record of the three models' completion files, so all three normalisations
come off the same forward passes. `acc_per_char` reproduces the stored
`accuracy` field in all 12 model x split cells, which is the check that the
recomputation is faithful.

n = 50 per split, chance 0.25, binomial SE about 0.061.

**`acc_per_char`** -- the repo's primary column:

| split | control | untaught | erased | erased - untaught |
|---|---|---|---|---|
| Rome QA train | 0.560 | 0.400 | 0.360 | -0.040 |
| Rome QA test | 0.420 | 0.260 | **0.400** | **+0.140** |
| Simdom train | 0.480 | 0.440 | 0.380 | -0.060 |
| Simdom test | 0.640 | 0.580 | 0.560 | -0.020 |

**`acc_uncond` (PMI ranking)** -- the column `EVALUATION.md` calls plausibly
right for an ablation study but the noisiest:

| split | control | untaught | erased | erased - untaught |
|---|---|---|---|---|
| Rome QA train | 0.480 | 0.320 | **0.460** | **+0.140** |
| Rome QA test | 0.460 | 0.340 | **0.460** | **+0.120** |
| Simdom train | 0.220 | 0.320 | 0.200 | -0.120 |
| Simdom test | 0.320 | 0.420 | 0.320 | -0.100 |

**`acc_raw`** is included only for completeness, since it is the retracted
column: Rome QA train 0.580 / 0.400 / 0.380, test 0.420 / 0.340 / 0.400.

### Which split the delta was chosen on, and why it matters here

Read from the code, not the config comment. `lment_pipeline.py:659` passes
**`train_items` only** into `search_deltas`; `test_items` is scored separately
at line 664 with `include_records=True` as the held-out report. The config's
note that "automatic delta selection needs all four" is a requirement that all
four splits be *present* (line 623 errors without test data), not that the
objective optimises on all four. So:

- **delta 200 was selected on** Rome QA train + Simdom train (50 + 50)
- **genuinely held out from selection:** Rome QA test + Simdom test (50 + 50)

Now re-read the accuracy contrast for erased - control with that in mind:

| split | delta fit on it? | accuracy delta | McNemar p |
|---|---|---|---|
| Rome QA **train** | **yes** | **-0.200** | **0.021** |
| Rome QA **test** | no | -0.020 | 1.000 |

**EMBER's accuracy efficacy lives almost entirely on the split its delta was
optimised on.** The objective maximised `1 - qa_retention` on QA train, and
that is exactly where the accuracy drop appears; on the held-out half the
accuracy effect is two questions out of fifty. That is overfitting to the
selection set, visible directly rather than inferred.

**This does not mean the erasure does nothing out of sample.** On the
continuous statistic the held-out half moves hard -- erased - control on Rome
QA test gives dz -1.031, p < 1e-5, and the held-out chunk loss (n = 3,000,
never touched by delta selection) is decisive. The honest statement is narrower
and sharper: *the erasure has a real out-of-sample effect, but its reported
accuracy efficacy is a selection-set artefact.*

**Consequence for pooling.** Rome QA train + test may be pooled to n = 100 for
the **twin** contrast, where nothing was fit on either half. They must **not**
be pooled for any **erasure** claim, because that mixes the selection set into
the held-out set. For the erasure, Rome QA test alone is the clean number.

Pooling for the twin contrast was checked rather than assumed, on three points:

1. **The halves are exchangeable in effect.** Welch two-sample on the
   per-question paired differences, train half against test half: p = 0.571
   (untaught - control), 0.641 (erased - control), 0.840 (erased - untaught).
   The effect is the same size in both halves, so pooling averages one thing.
2. **Unequal difficulty does not bias a paired test.** The halves are not
   equally hard -- the control scores 0.560 on train and 0.420 on test -- but
   every question is its own control across models, so difficulty cancels.
   Worth noting *why* the accuracy gap exists: the control's mean
   `gold_per_char` is nearly identical across halves (-0.8456 vs -0.8671), so
   it knows the gold answers about equally well in both, and the 14-point
   accuracy gap is about how the distractors happen to line up. One more view
   of accuracy being the noisier read of the same forward passes.
3. **The one large selection decision in this pipeline did not touch this
   bank.** The declarative-stem format was chosen on `stem_probe.py`'s own
   hardcoded 10 questions, which does not load
   `completion_questions.json`, so the 200-question bank is not a selection set
   for the format either.

What pooling actually buys: for **accuracy** on the twin contrast it takes n
from 50 to 100, which is the row in the power table above that needed 100 and
had 50 -- power roughly 0.5 to 1.0. For the continuous statistic both halves
already resolve at n = 50, so pooling only tightens an answer already in hand.
Pool the Simdom halves too; the specificity control deserves the same power.
Pooling Rome *with* Simdom remains wrong at any n -- keep them as strata, which
is what the `QA - SimdomQA` contrast does.

### The option-length confound, tested and ruled out

`gold_per_char` divides by the gold answer's character count, which is a crude
length correction, so it was worth asking whether the continuous effects track
answer length rather than knowledge. Pearson r between each question's paired
difference and its gold answer's character count, across all 12
contrast x split cells: **11 of 12 are non-significant.** The one that clears
0.05 is untaught - control on Rome QA train (r = +0.312, p = 0.023), which is
about what 12 tests produce by chance.

Specifically the cell that prompted the question -- erased - untaught on Simdom
test, the neighbour-domain effect at p = 0.0006 -- comes back **r = +0.114,
p = 0.427**. The length confound is not what is driving it. All twelve
correlations are mildly positive (+0.03 to +0.31), a weak systematic tendency
worth remembering, but far too small to manufacture effects of dz -0.35 to
-1.03.

### What this shows

**The accuracy columns cannot resolve the erased-vs-untaught contrast, and
under two of three normalisations they invert it.** On `pmi_per_char` (the
continuous statistic, gold answer only) the erasure overshoots the ablation on
both Rome halves. On accuracy, the erased model *ties the control* on Rome QA
test -- 0.400 against 0.420 -- while the untaught twin sits at 0.260, so read
naively the accuracy column says the erasure preserved Rome and never-training
destroyed it. Under PMI ranking the erased model ties the control on **both**
Rome halves (0.460 / 0.460).

McNemar exact on the discordant pairs says none of it is resolvable: erased vs
untaught gives p = 0.815, 0.167, 0.581, 1.000 across the four splits, on 13-19
discordant pairs. Erased vs control reaches p = 0.021 on Rome QA train and
p = 1.000 on Rome QA test -- the same contrast, the same model, two halves of
the same question set.

Two cells are outright incoherent and worth keeping visible: under PMI the
control scores **0.220 on Simdom train, below the 0.25 chance line**, and the
untaught twin *beats* it there (0.320). Nothing in the experiment predicts
that; it is the noise floor of a 50-question instrument on base models that
cannot really answer questions.

**This is not a side note about metrics, it is the mechanism of the overshoot.**
EMBER's delta objective scored specificity by accuracy, accuracy could not move,
so specificity read 1.00 at every delta and nothing ever opposed a larger edit.
The same coarseness that makes the table above unreadable is what licensed the
search to run to the end of its grid.

**Do not resolve this by picking the normalisation that agrees.** The reason the
held-out chunk loss exists is that it is the same question at n = 3,000 and
n = 5,004 on a continuous measure, and it answers unambiguously -- see the
RESULT section below. Where the two instruments disagree, the disagreement is
about statistical power, and the accuracy columns are the weaker instrument.

Also note EMBER's own sweep reported a Rome QA baseline of 0.48, which matches
neither split here (0.560 train, 0.420 test); it runs its own question protocol,
so its accuracy numbers are not comparable cell-for-cell with these.

## What this does NOT establish

**Not "no erasure can match the ablation."** This is the delta *EMBER's own
objective selected*. A weaker delta would plausibly overshoot less and spare the
neighbour; the sweep shows efficacy already at 0.52 by delta 10, where the edit
is far gentler. The honest claim is about EMBER as configured, with its own delta
selection -- not about the method's best achievable point.

Finding that point would mean scoring several deltas against the ablated twin and
keeping the closest, which is choosing a hyperparameter by the size of the effect
being measured. That is the failure that retracted two claims here. If it is
worth doing, the rule for picking the delta has to be fixed and written down
first, and it must not read the twin comparison.

**One concept, one pair, one seed, one cell.** Everything in `ROME_RESULTS.md`'s
scope caveats applies unchanged, plus: one of the 11 accepted cells, and one
feature within it.

**Input embeddings only.** EMBER edited 76 rows of `model.embed_tokens.weight`.
`lm_head.weight` is untied in these models and was not touched, while the
ablation shaped the whole network throughout training. The two interventions are
not the same kind of object, and that asymmetry is a reason to expect them to
differ -- it does not explain the direction of the difference, but it belongs
next to any claim that they should have matched.

Raw numbers: `results/completion/erasure_vs_twins.json`.
Erasure run: `Ember-on-LMEnt/grid/JUDGE_RESULTS.md`, job 867391.

## Artifacts on disk

Data, referenced by path rather than committed, as with `ember_eval/results/`.

| path | what |
|---|---|
| `/home/dcor/galbarak2/hf-models/lment-1b-rome-erased-b131k/` | the erased model, built by `materialize_erased_model.py` from job 867391's `erased_embeddings.safetensors`. Exactly 76 embedding rows differ from the control; the other 199 tensors and the untied `lm_head` are bit-identical. |
| `.../Ember-on-LMEnt/runs/Ancient_Rome_lment-1b-control-2e-b131k_20260908_045242/outputs/report.json` | the erasure run: delta sweep above, edited token ids, integrity check |
| `/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/completion/cmpl_erased2e_rome_*_870252.json` | the erased model's four Rome splits |
| `/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/completion/erasure_vs_twins.json` | the three contrasts above |
| `/home/dcor/galbarak2/lment-rome-check/results/ppl_erased2e_final_870356.json` | **pending** -- see the next section |

Jobs: erasure 867391; completion scoring 870252; held-out 870356 (running).

## RESULT: the held-out chunk loss (job 870356, analysed 2026-09-09)

Job 870356 COMPLETED (01:18:11, ended 2026-09-09T01:17:07) and is analysed
below. All three models scored the **same** 3,000 held-out and 5,004 control
chunk ids, from `rome_blacklist_sample3000.json` at seed 42 in float32 --
verified by reading `metadata` out of all three result files, not assumed.

    control  /home/dcor/galbarak2/lment-rome-check/hf/control-2e-step54832
    untaught /home/dcor/galbarak2/lment-rome-check/hf/norome-2e-step54832
    erased   /home/dcor/galbarak2/hf-models/lment-1b-rome-erased-b131k

**The control-vs-untaught run reproduces +0.2349 exactly**, so the instrument is
unchanged from `ROME_RESULTS.md` and the erased column is directly comparable.

| model | held-out (3,000) | control set (5,004) | its own gap |
|---|---|---|---|
| control twin | 2.5034 | 2.3769 | +0.1265 |
| untaught twin | 2.6939 | 2.3795 | +0.3144 |
| **EMBER-erased** | **3.7218** | **2.4125** | **+1.3093** |

Paired per-chunk differences of differences, all at label-permutation p = 0.0000:

| pair | held-out diff (dz) | control diff (dz) | **diff-of-diffs** |
|---|---|---|---|
| untaught - control | +0.2384 (1.052) | +0.0035 (0.085) | **+0.2349** |
| erased - control | +1.3793 (0.581) | +0.0332 (0.106) | **+1.3461** |
| erased - untaught | +1.1409 (0.512) | +0.0297 (0.098) | **+1.1111** |

### The prediction was confirmed, and by more than predicted

The 50-question result predicted the erasure overshoots the ablation by
1.7-2.2x. At n = 3,000 it overshoots by **5.73x** (+1.3461 against +0.2349).
This file's headline does not need revisiting; it needed a bigger instrument,
and the bigger instrument makes the conclusion harder, not softer.

**The control-set damage is real and was invisible at n = 50.** The ablation
moved the control chunks +0.0035 (dz 0.085, i.e. nothing). The erasure moves
them **+0.0332 -- 9.5x as much**, p = 0.0000 at n = 5,004. `dz` is still only
0.106, so this is a small effect per chunk; it is the *n* that resolves it. The
erasure damages general text. The MC splits could not have detected this.

### But it is not "the ablation, only stronger" -- the shape differs

This is the finding the diff-of-diffs summary hides, and it only appears in the
per-chunk distribution over the 3,000 paired held-out chunks:

| per-chunk diff | mean | sd | dz | median | p90 | p99 | max | >1 nat | >3 nats |
|---|---|---|---|---|---|---|---|---|---|
| untaught - control | +0.2384 | 0.227 | **1.052** | +0.172 | +0.413 | +1.357 | +2.04 | 2.2% | **0.0%** |
| erased - control | +1.3793 | 2.373 | **0.581** | +0.214 | +5.475 | +9.553 | +11.53 | 28.8% | **16.2%** |

Note the erasure has 5.8x the mean shift but a **smaller** `dz`. That is not
noise, it is the distribution: its **median chunk (+0.214) is barely different
from the ablation's (+0.172)**, while its p90 is 13x the ablation's and 16.2% of
Rome chunks get more than 3 nats/token worse -- a band the ablation never
enters at all, at any chunk (its worst single chunk is +2.04).

**Interpretation, offered as such:** the untaught twin degrades Rome text
*uniformly and mildly* -- the signature of a concept that was never learned. The
erasure leaves most Rome text roughly where the ablation does and *shatters a
subset of it*. Averaged into one number those look like the same intervention at
different strengths; per chunk they do not look like the same intervention at
all. Anything that reads the diff-of-diffs alone will miss this.

What this does NOT say: it does not identify which chunks shatter or why, and
16.2% is measured on one erasure of one concept in one model pair. The obvious
next question -- whether the shattered chunks are the ones EMBER's selected
features actually fire on -- is answerable from the artifacts already on disk
and has not been done.

Do NOT re-run the erasure at other deltas to improve this number -- see the
scope section above. The overshoot is now measured at n = 3,000; tuning delta
against it would be selection-on-the-outcome on the strongest instrument in the
project.


---

# PRE-REGISTRATION: the four-point delta-response sweep (2026-09-11)

**Written before the jobs were submitted, and before any result exists.** Jobs
are listed at the end of this section; if that list is empty, this section is a
plan and nothing below it has been measured.

## The rule this sweep sits against

The section immediately above says, in the project's own words:

> Do NOT re-run the erasure at other deltas to improve this number [...] tuning
> delta against it would be selection-on-the-outcome on the strongest instrument
> in the project.

That rule stands, and this sweep is run **under** it, not in spite of it. The
distinction it turns on:

* **Forbidden, and still forbidden:** running several deltas, finding the one
  whose diff-of-diffs sits closest to the ablation's +0.2349, and reporting that
  run as "the Rome erasure". That is tuning against the instrument.
* **What this is:** establishing whether the published Rome-vs-Baseball
  contrast is confounded by delta at all. The output is a curve, and the curve
  is the result.

Anyone who finds this section without the one above it should read that one first.

## Why the question is now live, when it was not before

`BASEBALL_RESULTS.md` (2026-09-11) did not exist when the rule was written. It
reports the same method, on the same control, through the same pipeline,
**undershooting** the ablation at 0.68x where Rome overshot at 5.73x. The two
runs differ in concept -- and also in delta, 200 against 10, which was never an
experimental choice: both came out of an objective whose specificity term reads
1.000 at every delta in both concepts and therefore never opposes a larger
delta. Rome's objective was still rising at 200, the last element of the grid.

So the headline cross-concept comparison **confounds concept with delta**, and
the delta was set by a metric this document already establishes is broken. That
confound cannot be resolved by measuring a third concept, because a third
concept would inherit the same broken selection. It is resolved by holding the
concept fixed and varying delta, which is this sweep.

## Design

Four runs, Ancient Rome, on the Rome pair's control (`lment-1b-control-2e-b131k`),
at `explicit_delta` 2 / 5 / 10 / 50. Configs
`configs/ember_lment_rome_erase_delta{2,5,10,50}_slurm.yaml` differ from the
published `ember_lment_rome_erase_nojudge_slurm.yaml` by **exactly two lines**
each -- `ember.explicit_delta` and `lment.slurm.job_name` -- verified by diff.
Same cell (rank 100 / sp 0.02 / seed 44), same judged feature 11 isolated by the
same `feature_ratio_threshold: 6.0`, same cached features (`reuse: true`; the
sparse fit does not reproduce across GPU models), same fp32, same partition and
constraint. Nothing but delta can explain a difference between these four.

10 is included because it is the value Baseball's own sweep chose; 50 because it
is the next published grid point above it; 2 and 5 to reach the region where the
edit is near or below exact removal (`delta 1` removes the concept component
exactly; `delta > 1` pushes the row into anti-concept space).

The grid is **not** extended above 200. Extending it would let the broken
objective pick a still larger delta, and the MC accuracy differences past delta
10 are one to two questions out of 50 against an SE of about +-0.067.

## Predictions, and what each outcome means

Read against the ablation (job 850249): held-out chunk-loss diff-of-diffs
**+0.2349 nats/token**, `dz` 1.052, control-set drift +0.0035, no single chunk
past +2.04.

1. **Some delta lands near +0.2349 with control drift near +0.0035, and the
   per-chunk distribution is uniform rather than tailed.** The 5.73x overshoot
   was a delta-selection artefact. The Rome/Baseball sign flip then says nothing
   about the concepts, and a third twin would not have revealed that. The
   finding becomes a statement about EMBER's objective, not about erasure.
2. **Rome overshoots at every delta that achieves any efficacy.** The sign flip
   is a genuine concept difference, and a third concept arm is worth its
   GPU-week.
3. **The mean tracks the ablation at some delta but the shattered tail persists
   at all of them.** Then uniform-versus-tailed is a property of the *mechanism*
   -- editing 72 embedding rows -- and not of its strength, which is the
   strongest version of the existing interpretation and does not depend on any
   delta being singled out.

`rel_edit_size` and `cosine` against `W_never - W_base` are reported per delta.
The prediction here is that **cosine stays near 0.001 at every delta**: delta
scales the step length along a fixed direction, so if the direction is wrong at
200 it is wrong at 2, and no delta makes the erasure approach the twin. If that
holds it is the cleanest result of the sweep, because it is the one quantity
delta cannot fix.

## Reporting rule, fixed now

Report **all four deltas**, in one table, including the ones that look bad. No
run from this sweep may be quoted as "the Rome erasure" and none supersedes the
delta-200 run in the sections above, which remains the published result because
it is the one the pipeline's own selection produced. If a later document needs a
single Rome erasure, it cites delta 200 and links here.

Jobs: submitted 2026-09-11 16:32-16:33 from c-002, all on n-601
(`killable`, a6000, account gpu-research -- unchanged from the published config).

| delta | job | config | run dir |
|---|---|---|---|
| 2 | `880400` | `ember_lment_rome_erase_delta2_slurm.yaml` | `runs/Ancient_Rome_lment-1b-control-2e-b131k_20260911_163315` |
| 5 | `880404` | `ember_lment_rome_erase_delta5_slurm.yaml` | `runs/Ancient_Rome_lment-1b-control-2e-b131k_20260911_163329` |
| 10 | `880387` | `ember_lment_rome_erase_delta10_slurm.yaml` | `runs/Ancient_Rome_lment-1b-control-2e-b131k_20260911_163218` |
| 50 | `880410` | `ember_lment_rome_erase_delta50_slurm.yaml` | `runs/Ancient_Rome_lment-1b-control-2e-b131k_20260911_163342` |

Run dirs are under `Ember-on-LMEnt/runs/` in the **LMEnt-initfix** checkout --
note the published Rome erasure (867391) ran from `LMEnt-ember` and Baseball's
(877959) from `LMEnt-mlp`, because `activate_env.sh` defaults `LMENT_ROOT` to the
`LMEnt` checkout and each run inherits whichever tree it was launched from. These
four were launched with `LMENT_ROOT` explicitly set to `LMEnt-initfix`.

Each run writes only `erased_embeddings.safetensors` (`save.full_model: false`).
Stage 2 -- materialise, held-out chunk loss, completion eval, weight comparison --
has NOT been run and is what actually answers the question above.

## Stage 1 result: the four erasures ran (2026-09-11 16:34)

All four `COMPLETED` in 1-2 minutes each -- far faster than the published runs
because no delta search executes when `explicit_delta` is set (`delta_search:
none` in all four reports, confirmed).

**Integrity, identical across all four:** `selected_feature_ids: [11]`,
`k_features_embed: 1`, `n_tokens_edited: 76`, `integrity.passed: true`, only
`model.embed_tokens.weight` in `changed_tensor_names`, `reload_state_match` and
`reload_logits_match` both true. The **76 changed embedding rows are the same 76
rows in every run** -- so across this sweep the support of the edit is fixed and
only its magnitude varies. That is the design claim, verified rather than assumed.

### EMBER's own MC metrics per delta, and the flaw reproducing

From each run's `report.json`. 50 questions per split, chance 0.25.

| delta | qa_retention (train) | efficacy (train) | **specificity** | objective (train) | efficacy (test) |
|---|---|---|---|---|---|
| 2 | 1.0000 | 0.0000 | **1.000** | 0.0000 | 0.0000 |
| 5 | 0.6522 | 0.3478 | **1.000** | 0.5161 | 0.1053 |
| 10 | 0.4783 | 0.5217 | **1.000** | 0.6857 | 0.1053 |
| 50 | 0.3913 | 0.6087 | **1.000** | 0.7568 | 0.2105 |

**`specificity` is 1.000 at every delta here too**, and `simdom_retention` is
exactly 1.000 in all four. This is the third independent reproduction of the
structural flaw -- Rome's published 8-point grid, Baseball's, and now this sweep.
The objective is efficacy alone, it rises monotonically with delta across the
whole range, and the delta-50 train objective (0.7568) sits right where the
published grid's monotone rise predicts between its 10 (0.686) and 200 (0.821).
**Nothing in the objective ever opposes a larger delta, at any delta.**

Read the efficacy column against the instrument, not past it: on test, deltas 5
and 10 are *tied* at 0.1053 and 50 reaches 0.2105 -- that is 2 questions against
4 out of 50, at a binomial SE of about +-0.06. `EVALUATION.md`'s standing
conclusion applies unchanged; these MC columns cannot rank deltas and are
recorded here only because the pipeline produced them for free.

**Stage 1 answers nothing about the twins.** It establishes that four erased
models exist, differ only in delta, and are internally sound. The question in
this section -- whether the 5.73x overshoot is a delta artefact -- lives entirely
in the held-out chunk loss, which has not been run.

## What the selection metrics actually compute, and where we deviate from EMBER

Added 2026-09-11 while the delta sweep was running, because the sweep's
`specificity: 1.000` column has now appeared in three independent runs and the
reason is algebraic rather than empirical.

### The definitions (`lment_pipeline.py:245-270`)

With `acc` the 4-option MC accuracy and chance 0.25:

    retention(edited, baseline) = clamp01( (acc_edited - 0.25) / (acc_baseline - 0.25) )

    qa_retention     = retention on the CONCEPT questions
    simdom_retention = retention on the SIMILAR-DOMAIN questions
    efficacy         = 1 - qa_retention
    specificity      = simdom_retention
    objective        = harmonic_mean(efficacy, specificity)

So `efficacy` is "the fraction of the model's **above-chance** concept accuracy
the edit destroyed" and `specificity` is "the fraction of the neighbour's
above-chance accuracy it left alone". Both are chance-corrected and both are
**clamped to [0, 1]**.

### Why specificity is 1.000 everywhere, and why that is fatal to the search

The clamp is the mechanism. `retention` is capped at 1.0, so **any** simdom
accuracy greater than or equal to baseline reads as exactly 1.000 -- the metric
cannot distinguish "untouched" from "improved", and it has no room below 1.0
until the neighbour's accuracy actually falls. On 50 questions it rarely does:
Rome's published sweep has simdom accuracy *rising* 0.38 -> 0.44 at deltas 5 and
10 and returning to 0.38 at 200, so it clamps at every row.

With `specificity` pinned at 1.0 the objective is not a trade-off at all:

    harmonic_mean(e, 1) = 2e / (1 + e)

which is strictly increasing in `e`. Verified against every erasure this project
has run -- Rome's published delta 200 (e 0.696 -> 0.821), Baseball's delta 10
(0.762 -> 0.865), and all four sweep cells (0.3478 -> 0.5161, 0.5217 -> 0.6857,
0.6087 -> 0.7568, 0 -> 0) -- reproducing the reported objective to four decimals
in all six cases.

**So the objective has been a monotone function of efficacy alone in every run,
and "maximise the objective" has been identical to "maximise delta" throughout.**
That is stronger than the empirical observation recorded above: it is not that
specificity *happened* to stay flat, it is that on this instrument it almost
cannot move, and when it cannot move the search has no opposing force by
construction.

### The delta grid is NOT EMBER's, and the deviation understates the overshoot

Upstream EMBER's default, in `ember/erasure/config.py`, unmodified since the
single `Import Ember code` commit `144d195`:

    _EMBER_DELTAS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

Every LMEnt config in this repo -- `ember_lment.yaml`, `ember_lment_slurm.yaml`,
`ember_lment_rome_slurm.yaml`, both erase configs -- instead specifies

    deltas: [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]

dropping 0.1, adding 2.0, and **truncating at 200 where EMBER continues to 500
and 1000**. The Rome config's comment that "the delta grid [...] is unchanged at
its published value" means unchanged from this repo's own base config; it is not
EMBER's grid, and that sentence should be read with this correction.

**This cuts in the direction of less overshoot, not more.** Rome's objective was
still rising at 200; on EMBER's real grid the search would have gone on to 500
and 1000 and chosen a more extreme edit still. The measured 5.73x is therefore a
**lower bound** on what stock EMBER does to this model, and the finding is
strengthened rather than weakened by the deviation.

Open, and a choice rather than a bug: whether to run the two missing grid points
(500, 1000) so that "what stock EMBER selects" is measured rather than inferred.
Not done, and not needed for the concept-vs-delta question this sweep asks.

### What is NOT wrong

`rank: 100` matches EMBER's own default. The retention/efficacy/specificity
formulas are upstream and unmodified. `ratio_thresh: 2.0` is not an upstream
*code* default -- `SelectionConfig.ratio_thresh` is `None` and documented as
SNMF's -- so the "published Gemma/Llama value" claim beside it comes from the
paper, which cannot be checked from the repo.

### Extension to EMBER's full grid: deltas 500 and 1000 (2026-09-11 16:54)

Added after the first four had run, so the reason is stated plainly: these are
the two grid points the repo's configs **drop** relative to upstream EMBER's
`_EMBER_DELTAS`, which runs to 1000. With them the sweep reaches EMBER's own
endpoint, so "what stock EMBER's objective selects on this model" becomes a
measurement rather than an extrapolation from a monotone trend.

The ground for adding them is **conformity to the published grid** -- fixed by
upstream code and visible before any of these runs existed -- not a response to
any twin comparison. No twin comparison had been computed for any delta when
these were queued; only EMBER's own MC metrics existed. The reporting rule is
unchanged and now covers six deltas: report all, quote none as "the Rome
erasure".

**Prediction on record before they ran:** the objective keeps rising through 500
and 1000, because with specificity pinned at 1.0 it is `2e/(1+e)` and can only
rise while efficacy does -- so stock EMBER would have selected **1000**, the grid
endpoint again, still without bracketing an optimum.

| delta | job | config |
|---|---|---|
| 500 | `880518` | `ember_lment_rome_erase_delta500_slurm.yaml` |
| 1000 | `880519` | `ember_lment_rome_erase_delta1000_slurm.yaml` |

Each differs from the published Rome config by the same two lines, verified by
diff against the config body.

### RESULT: the prediction above is REFUTED, and so is the "ran out of candidates" claim

Jobs 880518 (delta 500) and 880519 (delta 1000), both `COMPLETED`, both
`integrity.passed: true`, both feature 11, both 76 tokens -- same edit support as
the other four.

**Specificity leaves the cap.** It is not pinned at 1.000 everywhere after all:

| delta | QA acc (base 0.480) | Simdom acc (base 0.380) | specificity | objective |
|---|---|---|---|---|
| 2 | 0.480 | 0.380 | 1.0000 | 0.0000 |
| 5 | 0.400 | 0.440 | 1.0000 | 0.5161 |
| 10 | 0.360 | 0.440 | 1.0000 | 0.6857 |
| 50 | 0.340 | 0.420 | 1.0000 | 0.7568 |
| **500** | 0.360 | **0.340** | **0.6923** | 0.5950 |
| **1000** | 0.300 | **0.320** | **0.5385** | 0.6380 |

At 500 and 1000 the neighbour's accuracy finally falls below baseline, the clamp
stops binding, and the objective stops being `2e/(1+e)` -- it returns 0.5950 and
0.6380 against the 0.6857 and 0.8780 that formula predicts.

**The full-grid objective brackets an optimum, and the optimum is 200.**
Combining the published rows with these:

| delta | 0.5 | 1 | 2 | 5 | 10 | 50 | 100 | **200** | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| objective | 0.000 | 0.160 | 0.000 | 0.516 | 0.686 | 0.757 | 0.757 | **0.821** | 0.595 | 0.638 |

**Three corrections follow, and they run against what this document previously
asserted -- including what was added to it earlier today.**

1. **"200 is the last value in the grid with the objective still rising, so the
   search never bracketed an optimum; it stopped because it ran out of
   candidates" is refuted.** Extend the grid to EMBER's own endpoint and the
   objective *falls*. The search stopped at the argmax of the full published
   grid. That it also sat at the truncated grid's edge was a coincidence.

2. **The "lower bound" reading added earlier today is wrong.** It argued that on
   EMBER's real grid the search would have continued to 500/1000 and chosen a
   more extreme edit, making 5.73x an understatement. It would not have. **Stock
   EMBER on its own grid selects delta 200 -- exactly the value the published run
   used.** The grid truncation changed nothing about the selection, and the
   published Rome erasure is the one stock EMBER specifies.

3. **`objective = 2e/(1+e)` holds only where the clamp binds**, i.e. delta <= 200
   on this model. The metric is not structurally incapable of registering
   collateral damage; it is merely insensitive until the edit is an order of
   magnitude past the selected one. The earlier framing -- that nothing could
   ever oppose a larger delta "by construction" -- overstated it. What is true
   and survives: across the whole region any plausible delta lives in, specificity
   never moves, so the objective is efficacy alone there.

**The caveat that keeps this honest.** The argmax rests on differences of one to
three questions out of 50 at a binomial SE of about +-0.067: QA accuracy is
0.34 / 0.34 / 0.32 at deltas 50 / 100 / 200, and the specificity break at 500 is
simdom 0.38 -> 0.34, two questions. **The ranking among 50, 100 and 200 is not
statistically meaningful and 200 should not be called a real optimum.** What the
two new points do support, since both sit well below the peak and both move
specificity in the expected direction, is the qualitative claim that the
objective turns over somewhere beyond 200.

**What this does to the sweep's motivation.** The original framing -- Rome's 200
was an artefact of a truncated grid -- is substantially weakened; 200 is what
EMBER's own objective picks. The question the sweep actually answers is unchanged
and now the more interesting one: **does any delta reach never-having-learned?**
That is the held-out chunk loss, still running, and the Rome-vs-Baseball delta
difference (200 against 10) is now known to be a real difference in what the
objective selects rather than a grid artefact.

## The crossed 2x2: Baseball at delta 200 (2026-09-11 17:11, job 880580)

The delta sweep above attacks the concept-with-delta confound from one side only
-- Rome at many deltas, Baseball at one. The published erasures sit at different
deltas (Rome 200, Baseball 10) because EMBER's objective selected differently on
the two concepts, so **every Rome-vs-Baseball statement in `BASEBALL_RESULTS.md`
varies concept and strength simultaneously.**

Adding Baseball at delta 200 makes the design crossed:

| | delta 10 | delta 200 |
|---|---|---|
| **Ancient Rome** | job `880533` (this sweep) | job `867391` (published) |
| **Baseball** | job `877959` (published) | job `880580` |

With all four cells, concept and delta separate by difference instead of by
extrapolation: read Rome-minus-Baseball at each delta, and 200-minus-10 at each
concept. Config `ember_lment_baseball_erase_delta200_slurm.yaml`, two lines off
the published Baseball config, verified by diff.

**Pre-registration position.** Queued after the Rome sweep's erasures ran, so the
ground is stated plainly: **design symmetry**, completing a 2x2 whose other three
cells already existed. It is not a response to an outcome, and this was checked
rather than asserted -- at submission time `lment-rome-check/results/` contained
**no `ppl_rome_erased_d*.json` at all** and all six Rome chunk-loss jobs were
still running, so no twin comparison existed for any delta of either concept.
There was nothing available to select on.

**Prediction on record, before it ran.** Baseball's efficacy plateaus from delta
10 through 200 (`qa_retention` flat at 0.238), so on EMBER's MC instrument this
run should look nearly identical to the delta-10 run. The **chunk loss should
not**: the coefficient is 20x larger and chunk loss is unbounded. If the
metric-saturates-but-damage-does-not account is right, held-out damage should
rise steeply here while the MC columns barely move. **That dissociation -- same
MC numbers, very different text damage -- is the whole point of the cell**, and
it is the cleanest available test of why Rome overshot: not because Rome is
Rome, but because the objective could not see what delta 200 was doing.

Its chunk loss must be scored against the BASEBALL held-out sample, not Rome's:
`WORKDIR=/home/dcor/galbarak2/lment-baseball-check`,
`BLACKLIST=$WORKDIR/bb_blacklist_sample3000.json`.

### Baseball delta 200 (job 880580): the MC half of the predicted dissociation, confirmed

`COMPLETED`, `integrity.passed: true`, feature 10, **72 tokens edited** -- the
same edit support as the published delta-10 run, so again only the coefficient
differs.

| | delta 10 (published, job 877959) | delta 200 (job 880580) |
|---|---|---|
| qa_retention | 0.238 | **0.2381** |
| specificity | 1.000 | **1.0000** |
| objective | 0.865 | **0.8649** |

**Identical to four decimals at a 20x larger edit.** EMBER's MC instrument cannot
tell these two models apart at all -- which is exactly the plateau
`BASEBALL_RESULTS.md` recorded, now confirmed by running the endpoint rather than
inferred from the flat `qa_retention` column.

This is the prediction registered before the run, and it is the first half of the
dissociation. The second half -- whether held-out chunk loss rises steeply across
the same 20x, which it must if "the metric saturates but the damage does not" is
the right account of Rome's overshoot -- is job **880602**, scored against the
Baseball held-out sample.

Note what this already settles: **Baseball's delta 10 was not selected because
200 was worse on the objective.** The two are tied, and the `(objective, -delta)`
tie-break picked the smaller. Rome and Baseball did not land on different deltas
because the objective preferred different strengths; they landed differently
because Baseball's efficacy reached its ceiling inside the grid and Rome's did
not. The selection difference is an artefact of where a bounded instrument
saturates, not a judgement about the two concepts.

## The cross-concept test: damage follows the WORD, not the concept

Jobs 880679 (Rome-erased scored on Baseball) and 880680 (Baseball-erased scored
on Ancient Rome), completion eval, all four splits. These cells had never been
run: each erased model had only ever been scored on its own concept.

**On the mean, specificity looks perfect.** Rome-erased on Baseball: -0.000,
-0.004, -0.001, -0.013. Baseball-erased on Rome: zero to three decimals on all
four splits, with a per-question max |delta pmi| of 6.1e-05 -- floating-point
noise.

**Per question it is not perfect at all.** Rome-erased on Baseball:

| split | questions that moved | max delta pmi |
|---|---|---|
| QA train | 0 / 50 (max 5e-05) | -- |
| QA test | **1 / 50** | **-1.8913** |
| Simdom train | **1 / 50** | **-0.8471** |
| Simdom test | **1 / 50** | **-5.3537** |

Three questions out of 200, each shattered. The mean is ~0 because 197 are
untouched.

### The three are the same three, and the reason is a single token

    QA test       "...the minimum DISTANCE from home plate to the left or right field fence is"
    Simdom train  "The basketball position primarily responsible for scoring from DISTANCE is the"
    Simdom test   "In softball, the standard DISTANCE between bases is"

**`' distance'` is one of the 76 token rows Rome's erasure edited.** Decoded from
`changed_embedding_rows` against the control's `vocab.json`, the edited set is

    ' Rome', ' Roman', ' Romans', ' Caesar', ' BC', ' legion', ' consul',
    ' Nero', ' Julius', ' Marcus', ' Apollo', ' Pompe', ' Cic', ' dictator',
    ... and also: ' distance', ' rate', ' com', ' chaos', ' gods', ' priest',
    ' sacred', ' slave', ' slavery', ' enslaved', ' temples', ' baths',
    ' mythology', ' rhetoric', ' tyranny', ' Tables', ' Corpus', ' Circus'

Roughly a third of the 76 are **ordinary English words** that merely co-occur with
Rome, not Roman vocabulary.

The prediction this generates is exact and it holds: **every Baseball question
whose stem contains a swept-in word is damaged, and every damaged question
contains one. 3 hits, 0 misses.** (Matches on `' com'` in the scan are substring
artefacts -- "complete", "commits", "competition" tokenise to other tokens and
all show delta = 0.0000, which is itself a confirmation that the effect is
token-level.)

### What this settles

**The open question at the end of this document -- "whether the shattered chunks
are the ones EMBER's selected features actually fire on" -- is answered yes at
question level.** The trigger is not the concept. It is the presence of an edited
token, and the worst offenders are the ordinary words the feature swept in.

**Collateral damage is lexical, not semantic.** The erased model has not lost
knowledge of basketball or softball. It has a broken embedding for the English
word "distance", and every text containing that word pays for it. That is also
the most plausible account of the +0.0332 control-set damage (p = 0.0000) and the
16.2% of Rome chunks pushed past 3 nats/token: with ~25 common words among the 76
edited rows, a sixth of general Wikipedia text containing at least one of them is
entirely unsurprising.

**EMBER's specificity metric cannot see any of this, twice over.** Three
shattered questions out of 200 rarely flip a 4-option argmax, and the metric is
accuracy-based and clamped at 1.0 -- which is exactly what it reported.

**And it sharpens the headline comparison.** The untaught twin never learned
Rome. The erased model knows Rome about as well as before but has ~25 damaged
English words. Those are not the same kind of object, at any delta -- which is
the behavioural reading of the `cosine = 0.001` result.

Caveat: established at question level on n = 200 with 3 affected items. The
chunk-level version -- do the 16.2% shattered chunks contain edited tokens at a
higher rate than the rest? -- is still not run, and is the obvious follow-up now
that the mechanism is named.
