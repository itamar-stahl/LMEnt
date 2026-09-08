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

## UNFINISHED: the held-out chunk loss (job 870356)

Everything above rests on **50 questions per split**, which is why the `dz`
confidence intervals span about +-0.3 and why the residual on Simdom/test lands
at p = 0.057 rather than resolving. The held-out chunk loss is the same question
at **n = 3,000 held-out and 5,004 control chunks**, and it is the instrument that
produced the twins' headline. It was submitted, not yet analysed.

    job 870356, MODE=ppl, killable, --time=360, submitted 2026-09-09
    -> /home/dcor/galbarak2/lment-rome-check/results/ppl_erased2e_final_870356.json

Comparable to the twins by construction: `run_rome_heldout.slurm` hardcodes the
same `rome_blacklist_sample3000.json`, seed 42 and float32, and `match_by_length`
is deterministic given those, so all three models score the *same* chunk ids.

### What to do when it lands

`compare_heldout.py` is generic -- `--control`/`--ablated` are just model A and
model B, paired per chunk id -- so point it at the control and the erased model:

    python ember_eval/heldout_ppl/compare_heldout.py \
      --control /home/dcor/galbarak2/lment-rome-check/results/ppl_control2e_final_858234.json \
      --ablated /home/dcor/galbarak2/lment-rome-check/results/ppl_erased2e_final_870356.json

It prints `DIFFERENCE OF DIFFERENCES`. **The number to compare it against is
+0.2349 nats/token**, the ablation's, from `ROME_RESULTS.md`:

| | held-out (3,000) | control set (5,004) | its own gap |
|---|---|---|---|
| control twin | 2.5034 | 2.3769 | +0.1265 |
| ablated twin | 2.6939 | 2.3795 | +0.3144 |
| diff-of-diffs | | | **+0.2349** (dz 1.052 vs 0.085) |

The prediction from the 50-question result is that the erasure **overshoots**,
i.e. its diff-of-diffs exceeds +0.2349. If it does, at n = 3,000, that is a far
harder version of this file's conclusion. If it does not, this file's headline
needs revisiting -- the MC splits are the weaker instrument, not the stronger.

Read the control-chunk gap too: the ablation moved it +0.0035 (dz 0.085, i.e.
zero). If the erasure moves the control chunks materially, it is damaging general
text, which the MC splits could not have detected.

Do NOT re-run the erasure at other deltas to improve this number -- see the
scope section above.

