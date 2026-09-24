# Question-level distance to the concept-excluded twin

## What this measures

The paper ranks erasure methods by the absolute value of the **mean signed**
answer-NLL difference to the twin,

    | mean_i ( NLL_M(i) - NLL_T(i) ) |

That statistic cannot distinguish a model that matches the twin on every question
from one that is +2 nats off on half of them and -2 nats off on the other half:
the signed differences cancel and both score 0. The cancellation is not
hypothetical here — on Artificial Intelligence, RMU's signed mean is **-0.013
nats**, close enough to read as a near-perfect match, while its mean *absolute*
paired difference is **0.953 nats**, statistically indistinguishable from the
distance of the model that was never erased at all.

This analysis reports the two statistics that cannot cancel, both on the held-out
target questions only:

    D_abs(M,T) = (1/N) sum_i | NLL_M(i) - NLL_T(i) |
    D_abs(F,T) = (1/N) sum_i | NLL_F(i) - NLL_T(i) |     (the unerased full model)

    R_abs(M,T)    = D_abs(M,T) / D_abs(F,T)
    P_closer(M,T) = (1/N) sum_i 1[ |NLL_M(i)-NLL_T(i)| < |NLL_F(i)-NLL_T(i)| ]
    P_tied(M,T)   = (1/N) sum_i 1[ |NLL_M(i)-NLL_T(i)| = |NLL_F(i)-NLL_T(i)| ]

Exact ties are reported separately and are never counted as "closer". There were
none in any of the nine cells.

## How to read it

`R_abs` is a distance *relative to doing nothing*. The full model is the model
before any erasure, so it is the natural null: an erasure that does not move the
model toward the twin has no claim to having reproduced never-having-learned.

| `R_abs` | meaning |
|---|---|
| < 1 | the erasure model is closer to the twin than the full model is |
| = 1 | it is exactly as far away — the erasure bought no similarity |
| > 1 | it is **further** from the twin than the model that was never erased |

`P_closer` is the same comparison counted question by question rather than
averaged: the fraction of the 50 held-out questions on which the erasure model
lands nearer the twin than the full model does. 50% is the coin-flip null. It is
the robustness check on `R_abs`, which a handful of large-magnitude questions can
otherwise drive on its own.

The two can disagree, and the disagreement is informative. Rome's SNMF has
`R_abs` = 0.940 (a 6% improvement in average distance) but `P_closer` = 64% (it
wins on nearly two questions in three): many small wins and a few large losses.

## Data

- **Split: held-out `target_test` only**, 50 questions per concept, 150 in total.
  Selection-half questions, neighbouring questions, unrelated questions and SciQ
  are all dropped at load — the script filters on both the set name and the
  per-item `phase == "test"` flag, so a mislabelled set cannot leak.
- **NLL**: the teacher-forced mean over the correct continuation's answer tokens,
  in nats — the same per-question number the paper's tables use. The script
  re-reads `score_model.py`'s own records and checks `nll == mean(token_nlls)`
  for every model and every question before using anything.
- **Checkpoints**: the accuracy-selected winners, cross-checked at run time
  against `ember_eval/acc_selection/results_sciq/selected_checkpoints.json`. The
  run aborts if that file names a different checkpoint for any cell.
- **No model was loaded and nothing was re-scored.** The per-question NLLs come
  from the existing `report.py` output, scored 2026-09-20/21.

## Confidence intervals

Paired nonparametric bootstrap over the 50 questions: 10,000 resamples, seed 0,
question indices drawn with replacement, and **the same indices applied to the
method distance and the full-model distance in every draw**, so the pairing that
makes the ratio meaningful survives resampling. Intervals are percentile, 2.5th
and 97.5th. Tokens are never resampled independently; the per-question NLL is the
unit of analysis throughout.

## Files

| file | rows | what |
|---|---|---|
| `per_question.csv` | 450 | 3 concepts x 3 methods x 50 questions; every per-question NLL and both distances |
| `summary.csv` | 9 | one row per concept and method, with `R_abs`, `P_closer`, `P_tied` and CIs |
| `table.tex` | — | the ACL table; needs `booktabs`, no other package |
| `run_manifest.json` | — | commit, inputs and their SHA-256, checkpoints, metric definitions, bootstrap settings, all 122 validation checks |

Proportions in the CSVs are stored as fractions in [0, 1], not percentage
strings. `table.tex` is the only place they are formatted as percentages.

## Reproducing

    python ember_eval/nll_kl/question_level_similarity.py \
      --results-root /home/morg/NLP_2526b/galbarak2/runs/nll_kl/results_accwinners \
      --out-csv ember_eval/nll_kl/question_level_similarity/summary.csv \
      --out-tex ember_eval/nll_kl/question_level_similarity/table.tex \
      --bootstrap-repetitions 10000 \
      --seed 0

Run it from the repository root, with
`/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python` (the only
interpreter here that has numpy). `per_question.csv` and `run_manifest.json` are
written alongside the two named outputs; nothing is copied by hand. The script
prints all 122 validation checks and refuses to write anything if one fails.

## Data limitations found while building this

1. **`../export_20260922/per_item.csv` cannot be used for this analysis.** It
   predates accuracy-based checkpoint reselection and carries held-out
   `target_test` rows for only three of the nine selected checkpoints
   (`ember_ai_d500`, `rmu_rome_L6hi_a10`, `snmf_ai_ratio_in` — the three cells
   the reselection did not change). The other six appear there with
   selection-half rows only, because the Sunday grid sweep ran with
   `--only-phase selection`. The source of truth for all nine is
   `results_accwinners/<topic>/results.json`.
2. **The sibling `paper_tables/` directory is the *old*, NLL-rule selection**
   (`ember_rome_d5000`, `snmf_rome_ratio_in`, `ember_baseball_d500`,
   `rmu_baseball_L5mid_a100`, `rmu_ai_L5mid_a100`). Its `per_question.csv` has
   the right shape and the wrong six checkpoints. Do not join the two.
3. **One seed per cell.** No checkpoint here is seed-replicated, so a `R_abs`
   difference within the CIs should not be read as a method difference.
4. **N = 50 per concept.** The `P_closer` intervals are correspondingly wide
   (roughly +/- 13 points); only Baseball's EMBER and SNMF are separated from the
   50% null by the interval alone.
