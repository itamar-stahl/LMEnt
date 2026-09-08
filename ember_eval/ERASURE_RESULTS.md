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
