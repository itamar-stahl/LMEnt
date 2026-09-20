# lment-1b-baseball-erased-b131k — EMBER embedding erasure, NOT a control

**This model has edited weights.** It is `lment-1b-control-2e-b131k` with
`model.embed_tokens.weight` overwritten by an EMBER erasure of **Baseball** at
**delta 10**. It is *not* a twin, *not* a control, and *not* a pristine
checkpoint.

> Until 2026-09-19 this directory carried a byte-identical copy of the
> control's model card, which described it as "trained ... with nothing held out
> of the loss" and "a pristine reference copy". That card was wrong.
> `materialize_erased_model.py` copied every file out of the base directory,
> `README.md` included. Fixed there; this card replaces the copy.

| | |
|---|---|
| base | `lment-1b-control-2e-b131k` |
| concept erased | Baseball |
| delta | **10** |
| erasure job | `877959` |
| what differs from the base | the input embedding only — `lm_head.weight` is a separate, untied tensor and is carried across untouched |

Held-out TEST split: **QA 48% -> 36% (-12 points)**, SimdomQA completely flat (24% -> 24%, zero collateral). The `d200` variant gives the same -12 points, confirming `BASEBALL_RESULTS.md`'s pre-registered prediction that efficacy plateaus by delta 10. The standalone pipeline's own `report.json` shows its test-set evaluation was gated off entirely, which is why this erasure once looked unmeasured; it was scored directly with `score_ember_mc.py` instead.

## Erasure damage on these models is lexical

A chunk containing **none** of the edited token rows is damaged by exactly
zero — mechanically forced, its forward pass is bit-identical — and damage rises
monotonically with how many edited tokens it contains. This holds in
**non-concept text too**: for Rome, control-set text carrying 8+ edited tokens
is damaged +0.8547, **3.6x the entire Rome ablation effect**.

So these models have not "lost a concept"; they have broken token embeddings.
Read `ember_eval/ERASURE_RESULTS.md` before treating any of them as a
never-having-learned stand-in. Across deltas 2/5/10/50/200/500/1000 the
mean-ratio and median-ratio against the ablated twin cross 1.0 four to five
deltas apart and never together, so **no scalar delta reproduces the ablation**:
the two damage distributions differ in shape.

Whether erasure reaches never-having-learned is **subject-dependent**: on Rome
it overshoots (residual dz −0.315, CI excluding zero), on Artificial
Intelligence it lands on target. See `ember_eval/AI_RESULTS.md`.

## Provenance and rebuilding

Built by `ember_eval/materialize_erased_model.py` from the run's
`erased_embeddings.safetensors`, which verifies `base_config_sha256`,
`base_embedding_sha256` and `erased_embedding_sha256`, then re-reads the written
shard and requires the set of differing rows to equal `edited_token_ids`
exactly. Do not hand-edit; re-materialise instead.
