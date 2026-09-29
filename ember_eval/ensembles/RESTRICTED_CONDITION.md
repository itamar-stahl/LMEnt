# RMU+EMBER, restricted to cells that passed their sanity gate

Declared 2026-09-25, **before any of these three checkpoints is scored on a test
split**. The commit that carries this file precedes their held-out evaluation.

## Why a second condition is needed

The frozen grid's `RMU+EMBER` winners all failed `rotation_is_substantial`:

    rome      rmuember_rome_L5mid_a100      cos 0.041 -> 0.125   FAIL
    baseball  rmuember_baseball_L5mid_a100  cos 0.031 -> 0.076   FAIL
    ai        rmuember_ai_L5mid_a10         cos 0.038 -> 0.285   FAIL

That is not an accident of one cell. `H_selection` is the harmonic mean of
efficacy and preservation, EMBER has already supplied the efficacy, and an MLP
edit that does nothing inherits it while scoring perfect preservation. So the
rule prefers inert cells: on Baseball the top FIVE by H failed the gate, on AI
the top three.

The consequence is that the selected cells answer "what does this rule pick?"
and cannot answer "what does RMU contribute on top of EMBER?", because RMU never
engaged in them. Reporting only those would conflate a finding about the
selection rule with a finding about the method.

## The restricted condition

Same rule, same 50 selection questions, same tie-break order, with one
constraint declared here in advance: **the cell must pass its sanity gate**.
Ranking by `H_selection` among gate-passing cells only gives:

| Concept | Checkpoint | H_selection | rank in the unrestricted grid |
|---|---|---|---|
| Ancient Rome | `rmuember_rome_L6hi_a10` | 0.7438 | 2 |
| Baseball | `rmuember_baseball_L6hi_a10` | 0.6514 | 6 |
| Artificial intelligence | `rmuember_ai_L6hi_a10` | 0.5792 | 4 |

All three are `L6hi_a10` -- the identical configuration the standalone RMU round
selected for all three concepts. The restricted condition is therefore not an ad
hoc rescue: it is "the standalone RMU winner, applied on top of EMBER", which is
the comparison EMBER's own ensemble framing implies.

## What is and is not consulted

Selection uses the already-computed selection-half scores and the build-time
sanity gates. No twin and no test split is read in choosing these three. The
twins and test splits enter only in the evaluation that follows this commit.

## How this is reported

Both conditions were computed. The paper reports the rule-selected condition
only, and states in Section 5 that its cells failed the gate, so they measure
the selection rule rather than RMU's contribution.

The gate-restricted condition is **not** a reported row. `R_abs` and `P_closer`
exist for it -- `results/summary.csv`, method `RMU+EMBER-GATED`: 1.637 (Rome),
2.239 (Baseball), 2.468 (AI), each farther from the twin than the rule-selected
row it would sit beside. `R_KL` was never computed for these cells. Its
direction is given in the paper in one sentence, without a table row, because
the `R_KL` column would be empty.

The gap between the two is still a result -- it measures how far the
efficacy-preservation rule drifts from the method it is supposed to be tuning,
once another method has already done the erasing -- and the paper states that
gap in prose rather than in the table.

Decision recorded 2026-09-29, superseding the earlier plan in this file to
report both as rows.
