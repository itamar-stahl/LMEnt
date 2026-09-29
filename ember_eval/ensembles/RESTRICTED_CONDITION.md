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

## How both are reported

Both conditions appear in the paper, neither replaces the other:

- **`RMU+EMBER` (rule-selected)** -- what the paper's own selection rule picks.
  Its QA numbers are labelled as not evidence about RMU, because the cells
  failed their gate. It is evidence about the rule.
- **`RMU+EMBER` (gate-restricted)** -- what RMU contributes on top of EMBER when
  RMU actually engages. This is the primary ensemble number.

The gap between them is itself a result: it measures how far the
efficacy-preservation rule drifts from the method it is supposed to be tuning
once another method has already done the erasing.
