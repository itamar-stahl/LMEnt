# Prediction, recorded before any ensemble candidate is built or scored

Written 2026-09-24, before any `RMU+EMBER` or `SNMF+EMBER` model exists. No test
split, twin result or `H_test` for any ensemble cell had been produced when this
file was committed; the commit hash is the evidence. If the results contradict
what is below, that is the finding, and this file is not to be edited.

## The question this paper can ask and EMBER could not

EMBER's headline (Suslik et al. 2026, §4.3) is that ensembling an MLP method with
EMBER improves the efficacy–utility tradeoff, measured against the *pre-erasure*
model. This paper has a matched concept-excluded twin, so it can ask the strictly
harder question: does the ensemble move *toward never having learned the concept*,
or merely further away from having known it?

## Primary prediction (from the task specification)

Adding EMBER to an MLP method **improves `H_test`** — replicating EMBER's claim in
a base-model setting — **while moving the model further from the twin** in target
NLL and target KL, because it adds suppression to a method that already overshoots
on Baseball and AI.

## Competing prediction (recorded because this repo's own priors favour it)

Across four independent instruments this project has returned an RMU/SNMF null:
neither method moves target QA accuracy meaningfully, and on target NLL versus the
full model RMU and SNMF sit at +0.05 / +0.02 nats on Baseball and AI — nothing.
If the MLP methods contribute nothing standalone, the likeliest outcome is that
**each ensemble is statistically indistinguishable from EMBER alone** on every
axis: `H_test`, target NLL, target KL and `R_abs`. That is a null, not a failure,
and it is a stronger statement than the standalone nulls: it says the MLP methods
add nothing *even where there is something left to remove*.

The two predictions are distinguished by a single contrast — `RMU+EMBER` versus
`EMBER` alone at the same delta. Primary predicts a shift; competing predicts none.

## What would overturn the paper's headline

The paper currently concludes that the highest efficacy–preservation score does not
imply similarity to the twin. That survives unless an ensemble is **both** the
highest-`H_test` cell **and** the closest to the twin on target KL and on the
question-level absolute distance `R_abs` (`ember_eval/nll_kl/question_level_similarity/`).
If one cell is both, the abstract must change, and this is to be reported plainly
rather than argued around.

## Commitments

- Selection on the 50 selection questions only, `H_selection` per Eqs. 1-4.
- Twins and all test splits stay unread until the frozen-winner commit exists.
- No cell is re-picked after any test number is seen.
- A cell that fails `rotation_is_substantial` is reported as a failed gate, and its
  QA number is not evidence of anything.
- The ensembles were added in a later round than the original nine cells. The paper
  will say so.
