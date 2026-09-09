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

