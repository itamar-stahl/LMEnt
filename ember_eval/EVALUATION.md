# The 1B twin pair: what the evaluation found

Both twins were scored on all 3,600 EMBER multiple-choice questions -- 18
concepts x {QA, SimdomQA} x {train, test} x 50 -- by log-likelihood, in fp32, with
`run_twins_eval.slurm` (job `763384`). Identical questions, identical shuffled
option order, so every question is a matched pair.

EMBER's two subsets carry opposite predictions, and that is what makes them worth
keeping apart:

| subset | asks about | an ablation should |
|---|---|---|
| `QA` | the concept itself | lower the score -- efficacy |
| `SimdomQA` | the neighbouring domain (for Pornography: films, ratings, Hollywood) | leave it alone -- specificity |

## First pass: right/wrong. Nothing.

`paired_twins.py` runs McNemar's exact test per concept and subset. Pornography's
QA cell is the most lopsided in the table -- 7 questions the control got right and
the ablated twin missed against 1 the other way, p = 0.070, 32% against 26% -- but
it does not survive its own context:

- six untouched concept/subset cells moved at least as far;
- **the specificity control moved further than the concept itself** (SimdomQA
  b-c = +7 against QA's +6), which is the wrong shape for a real effect;
- the ablated twin is systematically worse everywhere, b-c mean +1.0 across the
  34 untouched cells.

Read as accuracy, the honest verdict is no detectable effect.

## Second pass: the log-likelihoods. The predicted pattern, at the resolution limit.

Accuracy discards almost everything the scorer computed. A question the control
gets right by 0.02 nats and the ablated twin gets wrong by 0.02 nats counts as a
full unit of evidence; a question where confidence in the truth collapsed by 3
nats but both still picked it counts as nothing. At 100 questions per cell, that
is what leaves the McNemar table too coarse.

The continuous values were already in the results JSON, so `loglik_twins.py` is a
reanalysis -- no GPU, no re-inference. Per-question paired differences, ablated
minus control, negative meaning the ablation hurt:

| statistic | what it measures |
|---|---|
| `logp` | log P(correct option) after a softmax over the four options. The smooth version of accuracy -- its argmax is exactly what the accuracy column reports. |
| `margin` | correct option minus best distractor. Signed distance from the decision boundary; its *sign* is the bit McNemar tests. |
| `optlp` | mean per-token log-likelihood of all four option texts, correct and distractors alike. Never looks at the answer key -- asks only how surprising the model finds text about the concept. |

`logp`, the primary. The ablated twin drifts -0.149 nats worse across all 3,599
questions, so the seventeen untouched concepts, not zero, are the null:

| column | Pornography | null mean | null sd | rank | rank p | t | t p |
|---|---|---|---|---|---|---|---|
| pooled | -0.334 | -0.139 | 0.131 | 3 / 18 | 0.167 | -1.45 | 0.083 |
| **QA** | **-0.573** | -0.161 | 0.155 | **1 / 18** | 0.056 | -2.57 | **0.010** |
| SimdomQA | -0.095 | -0.116 | 0.159 | 7 / 18 | 0.389 | +0.13 | 0.552 |
| **QA - SimdomQA** | **-0.478** | -0.045 | 0.174 | **1 / 18** | 0.056 | -2.42 | **0.014** |

`margin` agrees throughout: QA -0.656, rank 1/18, p = 0.018; QA - SimdomQA
-0.585, rank 1/18, p = 0.021.

Three things about this table matter more than the p-values.

**The rank test is floored.** With eighteen concepts the smallest attainable
one-sided rank p is 1/18 = 0.056, whatever the effect size. Rank 1 of 18 is the
strongest statement that test can make. The `t` column is the same comparison
without the ceiling -- one new observation against the seventeen, prediction-interval
form -- at the cost of assuming concept-level drift is roughly normal.

**The specificity control is flat, and it was not flat in the accuracy analysis.**
Pornography's neighbouring-domain questions sit at the null mean (rank 7/18) while
its own questions are the most damaged of any concept. Within the concept, that
gap is -0.478 nats at label-permutation p = 0.025. This is the shape an ablation
predicts, and the accuracy analysis reported the opposite shape.

**It is not one bad question.** Pornography's QA column ranks 1st of 18 on the
mean, 1st on the 20%-trimmed mean, 1st on the fraction of questions that moved
down (63%), and 2nd on the median. It replicates across EMBER's two disjoint
question sets independently: QA_train -0.728 (rank 2), QA_test -0.417 (rank 3).

## The probe that found nothing, and why that is interesting

`optlp` -- how surprising the model finds the option texts, ignoring which is
correct -- puts Pornography at rank 7 of 18, t = -0.36, p = 0.361. Gambling
(-0.50), Cannabis (-0.22) and Heroin (-0.15) all moved several times further than
the concept actually held out.

So holding 2,546 chunks out of the loss did not measurably raise the model's
surprisal on text about the concept. It shifted which of four options the model
prefers on questions about it. That is a discrimination effect, not a fluency
effect, and it is what the ablation should do: the removed chunks carry
associations (Kama Sutra <-> ancient Indian erotic text, Fanny Hill <-> first
English prose pornography), not the surface statistics of common words. The
option strings are short -- often one to five tokens -- which is also very little
text to detect a shift in.

## Verdict

Removing 0.024% of the corpus produced no difference in perplexity (13.53 against
13.51), no difference in accuracy, and a difference in the log-likelihood of
correct answers that is the largest of any concept, confined to the concept's own
questions, absent from its specificity control, robust to trimming, and replicated
across both question halves.

That is a consistent effect at p ~ 0.01 by a parametric test and p = 0.056 -- its
floor -- by the assumption-free one. It is evidence, not proof, and the honest
summary is that the effect is real if you accept the normality assumption and
suggestive if you do not.

## What would settle it

1. **More questions about the one concept.** 100 QA questions is the whole sample.
   Generating several hundred more Pornography questions in EMBER's format
   tightens the target's own estimate; it does not raise the rank-test ceiling,
   but it makes the t comparison rest on far less noise.
2. **More null concepts.** The ceiling is 1/18 because EMBER ships 18 concepts.
   Scoring a wider question bank -- any concept set, since the nulls only need to
   be untouched -- lowers the floor directly. Thirty-nine nulls would put 0.025
   in reach.
3. **A second ablated twin on a different concept.** The strongest design
   available: if the effect is real, each twin should show it on its own concept
   and not on the other's. That is a within-experiment specificity control that no
   amount of reanalysis can substitute for.

## Reproducing

    python3 loglik_twins.py                 # stdlib only, no GPU, ~1 minute
    python3 paired_twins.py                 # the accuracy analysis

Both read `results/twins4_{control,noporn}_763384.json`, which are gitignored --
they are 3.5 MB each and hold the per-question records the reanalysis needs.
