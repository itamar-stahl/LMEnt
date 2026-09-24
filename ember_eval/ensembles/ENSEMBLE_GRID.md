# Ensemble candidate grid and tie-break order

Committed **before any candidate is built or scored**, per Appendix A's selection
protocol. Appendix A's original guarantee — twins and held-out questions never
loaded during selection, tie-breaking fixed without consulting them — cannot be
recreated by default for conditions added after the first round's results are
known. It is reconstructed here deliberately: the grid and the tie-break order
are fixed in this file, in a commit that precedes every candidate.

Two new conditions, three concepts: **`RMU+EMBER`** and **`SNMF+EMBER`**.

## The EMBER base is the selected winner, not the released model

EMBER's stage system (§C.2) gives ensembles Stages 1+2+3+4: Stage 1 fixes EMBER's
δ in isolation by `H_score`, Stage 2 then sweeps the MLP grid on top. Stage 1 has
already happened in this paper — its outcome is the selected EMBER winner per
concept. The ensembles are therefore built on:

| Concept | EMBER base | path |
|---|---|---|
| Ancient Rome | `ember_rome_d200` | `<candidates>/ember_rome/d200/model` |
| Baseball | `ember_baseball_d10` | `<candidates>/ember_baseball/d10/model` |
| Artificial intelligence | `ember_ai_d500` | `<candidates>/ember_ai/d500/model` |

`<candidates>` = `/home/morg/NLP_2526b/galbarak2/runs/nll_kl/candidates`.

**This is a correction to the existing `mlp_erasure/run_post_ember_*.slurm`
scripts**, which point at the *released* models in `hf-models/`
(`lment-1b-{rome,baseball}-erased-b131k`). Those are not the selected winners:

- Rome — released and selected are the same object (target NLL vs twin agrees to
  1e-12). Harmless there.
- Baseball — differ at 1e-6; not the same scoring run.
- **AI — a different model entirely**: `+0.4728` vs `ember_ai_d500`'s `+1.6249`
  signed target NLL against the twin. Building the AI ensembles on the released
  model would silently answer a different question.

The scripts are repointed at the candidates tree. No EMBER checkpoint needs
producing; all three exist.

## Grids — matched to the standalone budget, not expanded

Rule §3.4: the ensemble grids are the standalone RMU and SNMF grids re-run on the
EMBER-erased base, so Appendix A Table 6 stays interpretable.

### RMU+EMBER — 8 cells per concept, 24 total (identical to standalone)

    layer  in {5, 6}
    band   in {mid, hi}
    alpha  in {10, 100}

The steering scale `c` is **not** a grid dimension: it is probed per concept from
the EMBER-erased model's own residual norms. It is never inherited from the
control's 72.7 / 83.2 / 93.8 — EMBER changed the embeddings, so the activation
distribution RMU's forget loss targets may differ, and a transferred `c` would be
meaningless. `rotation_is_substantial` gates are honoured; a cell that fails its
gate is reported as a failed gate and its QA number is not evidence.

### SNMF+EMBER — both readings, 10 + 10

Two variants, because they answer different questions and neither subsumes the
other. Each selects **independently within its own grid**, so the search budget
per reported condition equals the standalone budget. The winner is never taken as
the best across both variants — that would be a 2x search.

**Variant 2 — the headline.** SNMF re-derived from the EMBER-erased model
(`snmf.py factorize --model <erased>`). Protocol-faithful: EMBER's Stage 2 sweeps
the method's full pipeline on the ensembled model, and it is what a practitioner
holding an erased model would do.

    select in {ratio}          side in {in, out, both}      3 cells per concept

**Variant 1 — the appendix control.** Control-derived directions applied to the
EMBER-erased model, reusing the frozen factorizations already on disk
(`snmf_rome_883188`, `snmf_baseball_887805`, `snmf_ai_887806`). Holds the
intervention fixed and varies only the base model, so it separates *composition*
from *re-derivation*: if Variant 2 differs from standalone SNMF, Variant 1 says
whether that is the ensemble working or merely the factorization moving.

    select in {ratio}          side in {in, out, both}      3 cells per concept
    Rome additionally:         judge_both                   (see below)

**The headline designation is fixed here, in advance, on protocol grounds — not
by which variant scores better.** Both are reported whatever they show.

#### One asymmetry, recorded rather than smoothed over

Standalone Rome had a fourth SNMF cell, `judge_both`, which uses the Gemma judge.
Variant 2 runs ratio-only (`--skip-llm`), consistent with every re-run in this
series, because the judge is documented-flaky in this repo. So Variant 2 Rome has
3 cells where standalone Rome had 4. This *shrinks* the ensemble's search relative
to the standalone budget and so cannot inflate the comparison. It also costs
nothing that matters: the standalone Rome winner, `snmf_rome_ratio_out`, is a
ratio cell, so the analogue of the selected standalone configuration is present.
Variant 1 retains `judge_both` for Rome, since it reuses an already-judged frozen
factorization and introduces no new judge call.

### Totals

    RMU+EMBER            24 candidates   (8 x 3)
    SNMF+EMBER Var 2     9 candidates    (3 x 3)
    SNMF+EMBER Var 1     10 candidates   (3 x 3, + Rome judge_both)
    ----------------------------------------------------
                         43 candidates

## Selection rule — the paper's, unchanged

`H_selection` per Eqs. 1-4, on the **50 selection questions only**:

    Acc~   = clip01( (Acc - 0.25) / (Acc_full - 0.25) )        chance-corrected
    phi_eff = 1 - Acc~_target
    H      = HM( phi_eff, HM( Acc~_neighbour, Acc~_sciq ) )

Scored by `ember_eval/acc_selection/score_mc.py`, unmodified, so the new cells
travel the identical code path as the existing nine.

## Tie-break order — the paper's, unchanged

Applied only on an exact tie in `(H, preservation, efficacy)`, ascending:

| Method | key | order |
|---|---|---|
| RMU | `(layer, band, alpha)` | `band: mid=0 < hi=1`; lexicographic |
| SNMF | `(select, side)` | `select: ratio=0 < judge=1`; `side: in=0 < out=1 < both=2` |

These reproduce the keys recorded in
`ember_eval/acc_selection/results_sciq/selected_checkpoints.json` for the
standalone round (e.g. `rmu_rome_L6hi_a10` -> `(6, 1, 10.0)`).

## Freeze commitments

1. This file is committed before any candidate exists.
2. Selection uses the 50 selection questions only. The twins and every test split
   stay unread until the frozen-winner commit exists.
3. The winners are frozen in a commit of their own. Nothing is re-picked after any
   `H_test`, NLL, KL or `R_abs` number is seen.
4. A failed `rotation_is_substantial` gate is reported as a failure.
5. The paper will state plainly that the ensembles were added in a later round
   under this same frozen protocol.
