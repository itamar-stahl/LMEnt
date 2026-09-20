# lment-1b-ai-snmf-b131k — SNMF MLP erasure, NOT a control and NOT an EMBER model

`lment-1b-control-2e-b131k` with **MLP** weights edited by a semi-NMF feature
ablation targeting **Artificial Intelligence**. This is a different method from
everything else in this directory: EMBER edits the input **embedding**; SNMF
edits **MLP layers**. Do not group them.

| | |
|---|---|
| base | `lment-1b-control-2e-b131k` |
| method | SNMF (semi-NMF feature ablation) |
| selection | `ratio_only`, no LLM judge, tau 2.0 — **84 features** |
| delta in / out | **1.0 / 1.0** |
| layers edited | 4–6, both input and output side (3 layers each) |
| dtype | fp32 |

Full metadata is in `snmf_erasure_metadata.json` beside this file.

## delta 1.0 is the only value that erases, and that matters historically

`ablate_layer` applies `(I - delta*P)` onto a **unit** direction, so the targeted
component scales by `|1 - delta|`, not by `delta`:

    delta 1  -> 0     exact removal, the only true erasure
    delta 2  -> 1     sign flip, same magnitude
    delta 4  -> 3     TRIPLED
    delta 10 -> 9     9x

Every SNMF cell run in this project before 2026-09-13 used delta >= 4 — **every
one was an amplifying edit wearing the name of an erasure.** The write-ups that
called those a "null result" were reporting something real but not what they
said. This model is at the fixed delta of 1.0 and is a genuine test.

## It is a null, and it is the right kind of null

The SNMF AI arm moved `pmi_per_char` **upward** on all four splits — the wrong
sign for an erasure. Across three subjects (Rome, Baseball, AI), two
checkpoints, and three depth bands (4–6, 9–11, 14–16), delta=1 returns the
control's value or better, and permutation-of-neurons controls sit on top of the
real edit. The result is **two-sided**: these directions can be scaled 0x, 1.5x,
2x, 3x or inverted to −9x and concept QA does not respond in either direction, so
the model is not merely compensating for a removal.

Scope limit, stated honestly: this shows these *particular SNMF-identified
directions* are causally inert for this QA task. It does not show MLPs are
irrelevant — another feature-identification method could still find directions
that matter.

Three separate SNMF leads in this project have died on a seed check. **Standing
rule: no SNMF result counts until it reproduces across independent factorisation
seeds, and a permutation-of-neurons null is the right control for any
large-delta effect.** Full write-up: `mlp_erasure/DELTA_FIX_RERUN_RESULTS.md`,
and `ember_eval/AI_RESULTS.md` for this arm specifically.

## The weights are not here

Deleted 2026-09-18 to reclaim filer quota; see `WEIGHTS_REMOVED.md`. Config,
tokenizer, index, the erasure metadata and this card are intact, and the scored
results (`cmpl_aisnmf2e_ai_*`) are unaffected.

Rebuild with `sbatch lment-ai-check/snmf_ai_materialize.slurm`, which re-applies
the edit from the surviving factorisation `snmf_ai_887806` in minutes. **Do NOT
refit** — a sparse factorisation does not reproduce across GPU models at a fixed
seed, so refitting produces a different feature set under the same name. That
factorisation, not this directory, is the artifact that must not be lost.
