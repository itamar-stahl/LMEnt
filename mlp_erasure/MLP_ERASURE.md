# RMU and SNMF: making the two MLP erasure methods run on the LMEnt 1B

**Both scripts now run end to end on this project's models.** Neither had been
run here before. `snmf.py select` could not run at all -- it needed a Gemini
key this cluster does not have -- and both scripts loaded weights in a way that
would have made the resulting "erased" models unusable for the comparison they
exist to feed.

Branch `feature/mlp-erasure`, worktree
`/home/morg/NLP_2526b/galbarak2/LMEnt-mlp`, off `itamars/Ember-on-LMEnt`.

## Which two scripts, and why not the other pair

"The MLP erasure methods" here are `rmu.py` and `snmf.py` **in this directory**,
uploaded by Tamar on 2026-08-16 and identical on `main` and
`itamars/Ember-on-LMEnt`. They are written for these models -- OLMo-2, layers
picked from depth, `down_proj` looked up by name because WMDP's positional
`param_ids=[6]` misses it -- and `ember_eval/completion_eval/README.md` already
describes the project's erasure grid in their terms: RMU changes the selected
MLP `down_proj` matrices, SNMF changes `up_proj` and `down_proj`, EMBER changes
embedding rows, and there are EMBER+RMU and EMBER+SNMF cells.

There is a second RMU/SNMF pair, `Ember-on-LMEnt/ember/erasure/methods/{rmu,
snmf}.py`, registered in the EMBER fork's method registry. **Those were not
touched, and they are not one config away from running here.** Two reasons,
both structural rather than cosmetic:

- `SNMFMethod` reads its features through `features.ConceptContext`, which
  loads a potential-features CSV and SNMF pickles produced by the upstream
  `external/snmf` factorization for a specific model. No such artifact exists
  for the LMEnt 1B, and producing one is the job this directory's `snmf.py factorize`
  already does for itself.
- `RMUMethod` hardcodes `FIXED_PARAM_IDS = [6]` and Gemma/Llama layer settings,
  and the fork's LMEnt entry point (`ember/lment_pipeline.py`,
  `run_lment_ember`) is EMBER-only -- it never enumerates the other methods.
  `configs/` has `rmu_gemma.yaml` and `snmf_gemma.yaml` but no `*_lment*`.

If the intent was to route the MLP methods through the fork's grid/judge/eval
pipeline instead, that is a larger piece of work and worth saying so before
starting it.

## What was broken

### `snmf.py select` could not run — the judge

Appendix A.3's two-stage feature filter is two Gemini calls per feature.
There is no Gemini key on this cluster, so `select` had only `--skip-llm`, and
the comment on that flag says plainly that it stops matching A.3. That is the
whole selection step of the method.

The fix reuses what the EMBER fork already solved for *embedding* features: a
pinned local `google/gemma-4-12B-it` behind `ember.subprocess_judge.
SubprocessJudge`, run in its own interpreter because gemma4 needs newer
transformers than this pipeline. Its `describe_feature` / `classify_feature`
seams take a raw prompt and return text, and `classify_feature` already
normalises the reply to `{"is_member": ..., "confidence": ...}` -- exactly what
`_parse_membership` expected of Gemini. So `STAGE1`/`STAGE2` are unchanged and
the only thing that differs from the published recipe is which model answers
them. `--judge gemma` is the default; `--judge gemini` still works with a key.
MLP and embedding features are now interpreted by the same judge, at the same
pinned revision.

### Both scripts saved models that could not be compared to their control

Both loaded `torch.bfloat16`. `compare_weights.py` -- on `main`, at
`ember_eval/completion_eval/compare_weights.py`, not on this branch -- is how
this project measures an erasure against the twins at all. It reports
`D_erase = W_erased - W_base` per tensor against `D_target = W_never - W_base`,
and gives `rel_edit_size`, `cosine` and `progress_along_target` from them. It
is also how `ERASURE_RESULTS.md` can state that EMBER's erased model differs
from the control in exactly 76 embedding rows with the other 199 tensors
bit-identical.

A bf16 save makes `D_erase` non-zero in nearly every tensor, so it does not
merely weaken an integrity check -- it corrupts the metric. Measured on a
round-trip (`mlp_erasure/tests/test_mlp_erasure.py::SaveComparability`):

| save dtype | tensors differing from the source |
|---|---|
| fp32 | **2** -- exactly the two edited matrices |
| bf16 | **44 of 69**, embeddings and `lm_head` included |

Both scripts now default to `--dtype fp32`, which also matches the fp32
checkpoints on disk. `--dtype bf16` is still reachable.

I had also written that bf16 would lose the AdamW update itself. That is not
what happened: over 20 steps on a toy model bf16 reached 0.0142 relative
displacement of the edited matrix against fp32's 0.0149. It rotated the forget
activations about half as far, which is a reason to prefer fp32 and not
evidence about the 1B. The comparability argument is the one that stands.

### `device_map="auto"` with two models

RMU holds a frozen and an updated copy of the model and subtracts their
activations. `"auto"` is free to shard the two copies differently, and then
`mse_loss(r_act, r_ref)` gets its arguments on two devices and raises. Both
scripts now load onto one device (`--device`, default `cuda:0`); the retain
term moves its reference tensor explicitly as well. A 1B in fp32 is 4.4 GB, so
two copies fit on any card in `killable`.

### SNMF's layer ranges were literals from the wrong model

`--layers-in` / `--layers-out` defaulted to `(0,15)` and `(0,7)`. The LMEnt 1B
has **18** layers, so the in side skipped layers 16-17 and the out side covered
8 of 18. The fork's SNMF grid records the published Gemma-2-2B ranges as
in=`(0,25)` -- all 26 layers -- and out=`(0,8)`, its first third. The defaults
are now those two *fractions* of depth, which reproduce the 26-layer indices
exactly and give `(0,17)` / `(0,5)` here. Either flag still overrides.

### RMU's loss included pad positions

The forget term is an MSE against a fixed control vector over the whole padded
tensor, so part of every step's gradient went to dragging pad activations onto
the control vector, by a share that moved with how ragged each batch happened
to be. Both losses now mask, using the mask the cosine diagnostics already
carried.

### RMU's docstring had the model wrong

It described the 1B as 16 layers. `lment-1b-control-2e-b131k/config.json` says
18 layers, `hidden_size` 2048, `intermediate_size` 5632. The depth code was
already right; the documented band was not.

### Failures that looked like successes

- `snmf.py erase` with nothing selected printed "0 feature ablations total"
  and then **saved a model byte-identical to its control** and called it
  erased. It now refuses to save.
- `rmu.py --sanity` printed `FAILED: [...]` and exited 0. It now writes
  `sanity.json`, reports how far the edited matrix travelled
  (`edited_rel_change`), and exits non-zero.
- `snmf.py factorize` reported only the count of features clearing `tau` and
  discarded the distribution. `tau` gates this method the way
  `ratio_thresh: 2.0` gated EMBER's embedding run -- which died discovering
  that 2.0 sits *above* the maximum a 1B produces. Every factorization now
  writes `rho_stats.json` (max / p99 / p95 / median / min and counts at
  several thresholds) whether or not anything clears `tau`.
- `rmu.py` silently capped its step count. See the next section.

## A limitation to decide about, not a bug

`--max-num-batches` defaults to 150, the published value. The concept corpus
ships **300 sentences per concept** and `neutral_sentences.json` **300 in
total**, so at `--batch-size 16` there are 19 batches and RMU takes **19
optimiser steps, not 150**. That was invisible; it now prints a warning naming
the shortfall.

Nothing here can fix it. The options are a smaller batch size (300/150 = 2 per
batch), or harvesting more sentences -- Ancient Rome has 65,844 corpus chunks
and `Ember-on-LMEnt/sentences_gen` can harvest from them, but the *neutral*
side is a fixed 300 and is the binding constraint. **This is a scientific
choice about how far from the published grid to sit, so it is yours to make.**

## Verification runs

### What has been verified

Everything below was run, not reasoned about.

**Both pipelines, end to end.** On a 6-layer Olmo2 built from a config (same
architecture, tiny):

    rmu.py --probe -> rmu.py --sanity
    snmf.py factorize -> select --skip-llm -> erase -> verify

RMU's `--sanity` passes `forget_rotated`, `retain_preserved`,
`unlearn_loss_fell`, `spare_layer_frozen` and `edited_layer_moved` at 20 steps.
SNMF's `verify` prints its before/after table. Both write their artifacts.

**15 fast tests**, no checkpoint and no GPU, 0.2 s. They pin the two things
that were actually wrong about OLMo-2 -- that WMDP's positional `param_ids=[6]`
does not land on `down_proj` here, and that the bands and ranges have to come
from depth -- plus the fp32/bf16 save round-trip and the pad masking.

**The single-device fp32 load, on a real GPU.** 2.1 s for both scripts'
`load_model` on a login-node card. This matters because of the next section:
it is the evidence that `device_map={"": "cuda:0"}` is not what stalled.

**All three of `select`'s refusal paths**, each with a legible message:
`--judge gemini` with no key, `--judge gemma` with a bad `--fork-root`, and
`select` with zero candidates.

**The judge's snapshot and interpreter resolve.**
`resolve_judge_directory("google/gemma-4-12B-it", revision=707f0a3b...)`
returns the 9-file snapshot in `hf_cache/hub` with `local_files_only=True`, and
`conda_envs/gemma/bin/python` is present.

### Both scripts ran on the real 1B, 2026-09-09

**RMU probe, job 871246** (n-305, RTX 3090, fp32). Confirms on the real
checkpoint what the fast tests only pinned on a toy:

    params: {'arch': 'Olmo2ForCausalLM', 'down_proj_index': 8, 'positional_ok': False}
    18 layers, d_model=2048, depth-matched: [(4,[2,3,4]), (5,[3,4,5]), (6,[4,5,6])]
    layer 4: mean residual norm 72.7    layer 5: 83.2    layer 6: 93.8

`down_proj` is at **index 8**, so WMDP's positional `param_ids=[6]` -- which
`ember/erasure/methods/rmu.py` still hardcodes -- edits the wrong matrix on
OLMo-2. And **RMU's published steering grid transfers to this model**: against
a residual norm of 72.7-93.8, `{30, 100, 300, 1000}` is 0.4x, 1.2x, 3.6x and
12x the model's own scale, so the grid straddles the right range. That is the
opposite of what happened to EMBER's `ratio_thresh`, and it means choosing a
steering value here carries no selection-on-the-outcome risk. The rule fixed
in advance for the first real run: **the published grid value closest to 1x
the measured norm**, i.e. steering 100 at layer 5.

**RMU real run, job 871273** (n-301), steering 100 at layer 5 / layer_ids
[3,4,5], lr 1e-4, alpha 100, fp32. **All five sanity gates pass** and
`sanity.json` records:

    cos_forget_start 0.0173 -> cos_forget_end 0.0782   forget_rotated  true
    mean_cos_retain  0.9998                            retain_preserved true
    unlearn_loss_fell / spare_layer_frozen / edited_layer_moved  true
    edited_rel_change 0.01256                          failed: []

So RMU runs end to end and edits the matrices it claims to. **Read the numbers
rather than the booleans, though.** `forget_rotated`'s bar is
`end > start + 0.05` and the run cleared it by 0.011: the forget activations
moved from 0.017 to 0.078 cosine against the control vector, i.e. they are
still nearly orthogonal to it, and the edited matrix moved 1.3% in relative
norm. That is a real but **weak** intervention, which is what **19 steps
instead of the published 150** predicts. Nothing here says RMU erased Ancient
Rome; it says the method executes correctly, and the step-count ceiling in the
section above is the first thing between this and a meaningful erasure. No
model was saved -- `run_rmu.slurm` defaults to `--no-save-model`.

**SNMF factorize, job 871247**, layers 4/9/14, k=100, 300+300 sentences ->
8,370 concept and 8,564 neutral tokens, A = (5632, 16934), fp32. From
`rho_stats.json`:

| layer | max | p95 | median | n>2.0 | n>3.0 | stopped at |
|---|---|---|---|---|---|---|
| 4 | 4.99 | 3.32 | **1.05** | 9 | 6 | iter 1069 (converged) |
| 9 | 6.32 | 4.20 | **2.21** | 55 | 24 | iter 2999 (**hit the cap**) |
| 14 | 11.04 | 7.55 | **3.05** | 63 | 50 | iter 2553 (converged) |

**tau = 2.0 is comfortably reachable here**, which is the thing this run
existed to find out. EMBER's embedding ratio on this same checkpoint maxed at
1.6889 with nothing at all above 2.0; the MLP mass ratio clears it at every
layer. The published threshold transfers for this method even though it did
not for the embedding one, so SNMF will select features and produce a real
erasure.

**Read the second column before believing the fourth.** At layer 4 the median
feature sits at 1.05 -- neutral-balanced, which is what a discriminating
prefilter looks like -- and 9 of 100 pass. At layer 14 the *median* feature
carries 3x more mass on concept tokens and 63 of 100 pass. A prefilter that
admits two thirds of all features is not isolating the concept's features;
something inflates rho with depth.

The candidate explanation is the data rather than the code: `mass_ratio`
divides mean |Y| on concept tokens by mean |Y| on neutral ones, and the 300
Rome sentences are one topic while the 300 neutral ones are arbitrary
Wikipedia. A deep feature responding to topical homogeneity would score high
without having anything to do with Rome. **The test is a null concept** --
refit with one of EMBER's other 17 concepts and see whether layer 14 inflates
the same way. If it does, rho at depth measures homogeneity. `main` already
carries `cross_concept_null.py` and `NULL_CONCEPT_CONTROL.md` for this shape
of question. Until that is run, the judge stage is load-bearing rather than a
refinement, and layer 4 is the layer whose prefilter can be trusted on its own.

**Layer 9's row is an unconverged fit** -- it stopped because it reached the
`MAX_ITER=3000` this diagnostic set, not its patience criterion -- so its
55/100 should not be quoted beside the other two. Reconstruction error also
grows three orders of magnitude across these layers (1.43e10 -> 1.38e12),
tracking activation magnitude, so `recon` is not comparable across layers.

**Cost, measured.** The job took 29:36, of which about 26 minutes was the
checkpoint read; roughly 6,600 Semi-NMF iterations across three layers fit in
the remaining ~6 minutes, i.e. **18-36 ms per iteration**. A full 18-layer
sweep at the default `max_iter` 20000 is therefore **2-4 hours of arithmetic**,
not the ~20 minutes a 4 ms/iteration estimate suggested. It fits `--time=360`
with the load, but not comfortably at `--layer-batch-size 1`.

### What is NOT verified, and why

**Nothing has run against the real 1B checkpoint.** Four jobs were submitted to
`killable` / `a6000` -- the target EMBER's erasure of this concept used -- and
none got past loading the model:

| job | what | outcome |
|---|---|---|
| 870455 | `rmu.py --probe` | 40 min on "Loading checkpoint shards: 0/2", cancelled |
| 870456 | `snmf.py factorize`, layers 4/9/14 | 45 min on the same line, 4 of them as the node's only reader, cancelled |
| 870483 | judge smoke | cancelled at 18 min to free filer bandwidth for the other two |
| 870565 | `read_probe.slurm` | TIMEOUT, but it answered the question first -- see below |

All four ran on **n-602**, and `read_probe.slurm` says why. From that node:

| read of the 4.5 GB shard | rate | implied time for one shard |
|---|---|---|
| 300 MB, `iflag=direct` | 5.4 MB/s | 14 min |
| 300 MB, buffered | **1.5 MB/s** | **50 min** |
| the same file from the login node | 36.6 MB/s | 2 min |

**transformers loads buffered, so a shard read on n-602 really is about 50
minutes.** I had called this a wedged read rather than a slow one, on the
grounds that 45 minutes was too long for even the 7 MB/s this project has
measured on a contended node. That was wrong: the buffered path there is
another 4.5x slower again, and the jobs were reading the whole time. Measuring
it cost two minutes and would have saved an hour of cancelled jobs.

Nothing was wrong with the card, the job, or the code -- the same
single-device fp32 load path completes in 2.1 s on a login-node GPU.

**Gal's call, 2026-09-09: drop `--constraint=a6000` and take any card in
`killable`.** The constraint was inherited from EMBER's Rome config, where its
stated purpose was staying out of the h100 pool the twin training needs, and
every other card in `killable` does that equally. Both SLURM scripts now carry
no constraint. `run_rmu.slurm` additionally excludes n-202..205: those 2080s
have 11 GB, and RMU is the one stage holding two fp32 copies of the model at
once, so 2 GB of headroom for the autograd graph is too thin to rely on.

Resubmitted on that basis as jobs **871246** (RMU probe), **871247** (SNMF
factorize, layers 4/9/14) and **871248** (read probe, to record what the new
node serves).

### The Gemma judge answered both stages, 2026-09-09

Job **871285** ran `MODE=judge-smoke` to completion on **n-301**, which was the
last unexercised code path in `snmf.py select`. Both real prompts came back
correctly parsed:

| feature | STAGE1 description returned | `is_member` | confidence | accepted |
|---|---|---|---|---|
| Rome tokens (`Rome`, `Caesar`, `Senate`, `legion`, `Augustus`, ...) | "the historical and cultural elements of Ancient Rome" | `true` | 0.99 | yes |
| units of measure (`kilometre`, `hectare`, `acre`, `tonne`, ...) | "various units of measurement for physical quantities" | `false` | 1.00 | no |

Both stages produced parseable JSON, `raw_stage2` came back well-formed in both
cases, and the accept/reject decision matched the expectation written into the
smoke test before it ran. `--judge gemma` is therefore exercised, not just
plausible, and `select` no longer has to fall back to `--skip-llm` -- the
fallback whose own comment says it stops matching Appendix A.3.

Worth recording for card selection: it ran on an **RTX 3090 (24576 MiB)** and
took roughly 35 minutes wall for the 23.9 GB read plus four generations. A 12B
judge does fit on a 24 GB consumer card. It does *not* fit on the 11 GB 2080s
at n-202..n-205, and nothing had prevented 871285 from landing on one -- so
`run_snmf.slurm` now excludes them for every mode and additionally fails fast
in under a second if it finds less than 20 GB of VRAM, rather than discovering
it 15 minutes into the read.

### Parameters for the select/erase/verify chain, fixed before it ran

Written down here, and committed, **before** job submission, because
`ERASURE_RESULTS.md` is explicit that choosing an erasure hyperparameter by the
size of the effect it produces is the failure that retracted two `acc_raw`
claims.

**The rule: every parameter takes its published value. None is tuned against
this model's output.**

| parameter | value | source |
|---|---|---|
| `tau` | 2.0 | the paper's threshold, already used at factorize time |
| ~~`--delta-in`~~ | ~~4~~ | **RETRACTED -- see "The delta rule was wrong" below** |
| ~~`--delta-out`~~ | ~~4~~ | **RETRACTED -- see "The delta rule was wrong" below** |
| judge | `gemma` two-stage | the only judge reachable here; substitution already disclosed |
| layers | 4, 9, 14 | whatever 871247 factorized -- see the caveat below |

The layer set is the one place where this chain is a **pipeline verification and
not a candidate result.** Layers 4/9/14 were chosen for the factorize as a
diagnostic spread across depth, and only layer 4 falls inside the paper's
22-34% depth band (`layers_by_depth(18)` gives 4-6). A publication-grade
erasure should re-factorize on 4/5/6; that costs another 2-4 GPU-hours at the
measured 18-36 ms/iteration and is a scientific choice, so it is not folded
into this run. Read what follows as "the four stages execute and produce
coherent artifacts on the real checkpoint", not as "this is how much Rome SNMF
removes".

One thing the chain will test that nothing else has: whether the ratio
prefilter's behaviour at depth survives the judge. `rho_stats.json` records
63/100 features clearing tau=2.0 at layer 14 against 9/100 at layer 4, and I
flagged that the likely confound is the data -- 300 topically homogeneous Rome
sentences against 300 arbitrary neutral ones -- rather than layer 14 genuinely
carrying seven times as much Rome. If the judge rejects most of layer 14's 63
while keeping most of layer 4's 9, that is evidence the prefilter is loose at
depth and the judge is doing the real work. If the judge accepts layer 14's
wholesale, the confound is still live and the null-concept refit is the test
that settles it. Either way this is a diagnostic reading of one run, not a
measurement.

### The judge selected 13 of 127, and refuted my prediction (871388)

Job **871388** COMPLETED in 45:14 on n-301, judging all 127 candidates from
871247 through both evidence sources (activation and projection, accepted by
either independently). Selection metadata records
`selection_method: ratio_then_gemma`, `confidence_threshold: 0.85`,
`judge_top_tokens: 20`.

| layer | candidates (rho > 2.0) | selected by judge | accept rate |
|---|---|---|---|
| 4 | 9 | **0** | 0.0% |
| 9 | 55 | 6 | 10.9% |
| 14 | 63 | 7 | 11.1% |
| **total** | **127** | **13** | **10.2%** |

**I predicted the wrong thing and the record should say so.** The pre-committed
diagnostic read: "if the judge rejects most of layer 14's 63 while keeping most
of layer 4's 9, that is evidence the prefilter is loose at depth." The opposite
happened at layer 4 -- all nine of its candidates were rejected -- and layers 9
and 14 accept at rates indistinguishable from each other (10.9% vs 11.1%).

What that does and does not settle:

- **The judge is doing real work.** It rejects 90% of what the ratio prefilter
  passes, so tau > 2.0 alone is not a concept filter at this scale.
- **It does not resolve the depth confound.** Acceptance is flat across layers 9
  and 14, so layer 14 still contributes the most selected features purely
  because it had the most candidates. The question of whether rho at depth
  tracks Ancient Rome or the topical homogeneity of the 300-sentence probe set
  is still open, and the null-concept refit is still the test for it.
- **Layer 4 producing zero is itself informative.** Its nine features cleared
  tau but none read as Ancient Rome to the judge. At 22% depth these may be
  lexical or syntactic rather than semantic. One layer, one concept, so this is
  a lead rather than a result.

### That result exposed a fourth silent-success path, now fixed

The selected features live only at layers 9 and 14. `default_layer_ranges(18)`
gives `layers_in = [0,17]` and `layers_out = [0,5]`, so:

    layer  4:  0 features -> skipped (no features selected)
    layer  9:  6 features -> delta_in=4.0  delta_out=0.0
    layer 14:  7 features -> delta_in=4.0  delta_out=0.0

    total feature ablations = 13
    layers edited input-side (up_proj)  : 2
    layers edited output-side (down_proj): 0

**`erase` would have applied `delta_in` at two layers, `delta_out` at none, and
saved the result as an SNMF erasure.** SNMF projects the directions out of both
matrices; this is half the method. `total` was 13, so the `total == 0` guard
stayed quiet -- it counts features, and the features were all there. Nothing
downstream could have told, and `compare_weights.py` would have happily
reported a `D_erase` for it.

This is the same family as the three silent-success paths already fixed, found
the same way: by computing what the run would do before running it.
`check_both_sides_applied` now refuses the case, naming the layers that hold
features and the range that excludes them:

    erase: --delta-out=4.0 was requested but the output-side (down_proj) edit
    reached NO layer, so the result would be a one-sided erasure saved under
    the method's name.
      layers holding selected features: [9, 14]
      --layers-out range in effect:     [0,5]
    They do not intersect. Either factorize a layer inside [0,5] and select
    features there, widen the range with --layers-out lo hi, or pass the delta
    for this side as 0 to say the one-sided edit is intended.

A deliberate `--delta-out 0` still passes -- only asking for a side and
receiving nothing is an error. `verify` carries the same check, so it cannot
report activation drops for an edit that was never applied on one side, and
`snmf_erasure_metadata.json` now records `layers_edited_input_side` and
`layers_edited_output_side`. Four tests pin it (19 total, 12.4 s).

**Consequence for the erasure itself: it is blocked on a scientific choice, not
on code.** To erase with both sides, a layer inside `[0,5]` must hold selected
features. Layer 4 is the only band layer factorized and its candidates were all
rejected, so the options are to factorize layers 5 and 6 and see whether the
judge keeps anything there (2-4 GPU-hours), or to widen `--layers-out`, which
departs from the published depth fractions. Both are decisions about the
experiment rather than fixes, so neither is taken here.

### The depth band does carry Rome features (871493, 871547)

Gal's call after the layer-4 result was to test the rest of the band. Job
**871493** factorized layers 4, 5, 6 in 29:36; job **871547** judged all 56
candidates in 37:43.

| layer | candidates (rho > 2.0) | selected | accept rate |
|---|---|---|---|
| 4 | 9 | **0** | 0.0% |
| 5 | 34 | 3 | 8.8% |
| 6 | 13 | 3 | 23.1% |
| **total** | **56** | **6** | **10.7%** |

**Layer 5 is inside the output band `[0,5]`, so both sides are now reached** and
the one-sided guard passes: layer 5 gets `delta_in` and `delta_out`, layer 6
input-side only, layer 4 skipped. The erasure is no longer blocked.

Two checks worth recording. **Layer 4 reproduced exactly** -- 871493's rho
statistics for it are identical to 871247's to three decimals on max, p95,
median and count, both on n-301. So the factorization is deterministic on a
fixed card, and the known instability is across GPU *models*, not general
nondeterminism. And **the judge's acceptance rate is stable across disjoint
candidate sets**: 10.2% on the first 127, 10.7% on these 56. Layer 4 returning
zero twice is now a repeated observation rather than a fluke.

### verify at the published delta says the edit AMPLIFIES (871607)

    layer  concept before   after     drop    neutral before   after    drop
        5         48.48    160.3  -230.7%             13.96   29.26  -109.6%
        6         50.56    119.7  -136.8%             18.92   32.89   -73.8%
    mean concept drop -183.8%, mean neutral drop -91.7%

Activation went **up** roughly 3x. This is arithmetic, not a bug.
`ablate_layer` applies `(I - delta * P)` on the feature's support, so the
component along the feature scales by `|1 - delta|`, and `verify` takes
`.abs()` of that component. Its reported drop is therefore `1 - |1 - delta|`:

| delta | component | reported drop |
|---|---|---|
| 1 | removed | +100% |
| 2 | sign flipped, magnitude kept | 0% |
| **4** | **magnitude x3** | **-200%** |

Predicted -200%, observed -183.8%; the gap is the coverage mask and layer 5's
edit feeding layer 6. Confirmed numerically on a toy model: after/before came
to 0.5147 / 0.1410 / 1.0000 / 2.9734 at delta 0.5 / 1 / 2 / 4 against
`|1-delta|` of 0.5 / 0 / 1 / 3. Four tests pin it.

The script's own advice here was wrong twice over and is fixed: it said "the
edit is not doing much, raise delta", which moves further from zero, and it
read the neutral rise as evidence of poor specificity when the scaling applies
to everything in the support and says nothing either way.

### The mechanism works and is selective, at delta = 1 (871613)

Run as a **code-correctness diagnostic, not a delta search** -- delta 1 is the
unique value at which `(I - delta*P)` is an exact projection, so it answers
"does this remove anything at all, and is what it removes concept-specific?"

    layer  concept before   after    drop    neutral before   after   drop
        5         48.48    23.11   52.3%             13.96   10.12  27.5%
        6         50.56    23.77   53.0%             18.92   14.67  22.5%
    mean concept drop 52.7%, mean neutral drop 25.0%, selectivity +27.6%

**Concept activation falls 52.7% while neutral falls 25.0%** -- the edit removes
about twice as much concept as neutral, and no diagnostic fires. The machinery
is sound: selection identifies features the ablation can actually reach, and
what it reaches is concept-biased rather than generic.

(It is not 100% because only 6 of 100 features per layer are removed and the
coverage mask at gamma 0.95 keeps a subset of neurons. 52.7% from six features
is a large effect, not a weak one.)

### What is now open, and it is not a code question

`delta` and the verify metric are mutually inconsistent, and **only one of them
can be right**:

- if the published `delta 4` is correct for this update rule, then an
  `.abs()` drop can never be the check -- the design reverses and amplifies the
  direction, the way RMU's misdirection does, and success has to be measured
  some other way;
- if a drop is the goal, `delta` must be below 2, and 1 is the exact-removal
  point.

**This is not mine to settle**, and it is exactly the kind of choice
`ERASURE_RESULTS.md` says must be fixed by a written rule before it is made,
not after seeing which value produces a nicer number. Both results above are
recorded so that whichever rule is chosen, the other value's outcome is already
on the page and cannot be quietly dropped.

**No model has been saved.** `erase` was not run, because at delta 4 it would
write a checkpoint whose selected features are amplified threefold, and at
delta 1 it would be using a value nobody has yet chosen.

## The delta rule was wrong, and it shipped

**An erasure run at `--delta-in 4 --delta-out 4` produced a model that scored
BETTER on the erased concept than its control.** That is not a failed erasure.
It is the opposite intervention, saved under the method's name.

**Cause.** `ablate_layer` applies `(I - delta*P)` where `P` projects onto a
**unit** direction, so the targeted component scales by `|1 - delta|`, not by
`delta`. The table was already in this file and in `cmd_verify`'s comments, and
the `DeltaScalingLaw` tests already pinned it -- but it was read as a quirk of
the *verify metric* rather than as a statement about the *edit*:

| delta | component becomes | |
|---|---|---|
| 1 | 0 | the only exact removal |
| 2 | sign flipped, same size | erases nothing |
| **4** | **x3** | **what was run** |
| 7 / 10 | x6 / x9 | |

Only `0 < delta < 2` shrinks anything.

**Why the rule looked right.** The section above fixed "every parameter takes
its published value" and read 4 off `configs/snmf_gemma.yaml`. But that file
says `in_deltas: [1.0, 4.0, 7.0, 10.0]` -- delta is a **swept dimension**, not a
value. `SNMFMethod.enumerate_hps` crosses all four with all three layer ranges,
and `_run_method_grid` then throws away the cells that fail `max_qa_acc: 0.6`
and `min_mmlu: 0.7`. The amplifying cells exist in the reference precisely
because the downstream eval deletes them. Lifting one out of the sweep and
saving it removed the only thing protecting against it. Taking "the published
value" for a swept parameter is a category error, and it is the same shape as
the `acc_raw` failures `ERASURE_RESULTS.md` retracted.

**Two defaults made it the path of least resistance.** `snmf.py erase` and
`snmf.py verify` both defaulted `--delta-in` and `--delta-out` to **4.0**, so
the invocation documented below produced an amplifying edit without anyone
passing a delta at all.

**Fixed.**

- `erasure_scale(delta)` returns `|1 - delta|`, and `check_delta_erases` runs
  immediately before anything is written. A delta that does not shrink the
  component is refused, naming the factor and the flag.
- Both defaults are now **1.0**, the unique exact-removal point. `erase` and
  `verify` take the same defaults and the same gate, so verify cannot report on
  a cell erase would refuse.
- `--allow-amplification` reaches the higher cells **deliberately**, for a real
  sweep selected on a downstream eval. It warns, and
  `snmf_erasure_metadata.json` now records `erasure_scale_in`,
  `erasure_scale_out` and `amplification_allowed`, so an amplifying checkpoint
  can never be read back later as an erasure.
- A fifth silent-success path, found while fixing this: `cmd_erase` incremented
  `applied_in` / `applied_out` from the delta alone, so a layer whose features
  all lost their support to the coverage mask still counted as having covered
  that side, and `check_both_sides_applied` passed. It now counts only layers
  that actually edited something (`bool(d_in and n)`).

**What still is not decided.** `delta = 1` is the value at which the update is
an exact projection; it is *not* a claim that 1 is the right erasure strength.
Choosing that is the sweep-and-select the reference does, and it needs the MC
eval, not this script. What has changed is that the range `[2, 10]` can no
longer be entered by accident or by default.

### RMU in the fork edited the wrong matrix

`ember/erasure/methods/rmu.py` hardcoded WMDP's `FIXED_PARAM_IDS = [6]`, and
`rmu/utils.py:get_params` selects parameters **by position** inside a decoder
layer. On OLMo-2, `self_attn.q_norm` and `self_attn.k_norm` sit ahead of the
MLP and shift everything by two, so index 6 is `mlp.gate_proj` and `down_proj`
is at **8** -- which job 871246 already measured on the real 1B
(`down_proj_index: 8, positional_ok: False`). RMU is defined to move
`down_proj`; with the published index it optimised `gate_proj` and left
`down_proj` bit-identical.

Our `rmu.py` was never affected -- it looks the matrix up by name. The
fork's copy now resolves the index from the loaded model
(`_down_proj_param_ids`) and logs when it disagrees with WMDP's constant.

### Tests

`mlp_erasure/tests/test_erasure_direction.py`, 24 tests, no checkpoint and no
GPU. The suite is now **77 tests, 0.7 s**.

The load-bearing one measures neither a weight statistic nor a `down_proj`
input activation -- both of those see only one side of the edit. It hooks the
MLP output and measures how much of the feature direction the layer writes into
the **residual stream**, which is what a downstream MC eval actually responds
to. Against a 6-layer OLMo-2:

| delta | residual-stream concept contribution |
|---|---|
| 1.0 | **0.35x** -- erased |
| 4.0 | **2.26x** -- amplified |
| 7.0 | 7.75x |
| 10.0 | 16.62x |

and driving the real CLI end to end: `--delta-in 4 --delta-out 4` exits 1 and
writes no checkpoint; the defaults write one whose concept contribution is
0.35x its control; `--allow-amplification` reaches 4 and records 2.26x plus
`erasure_scale_in: 3.0` in the metadata.

## How to run it

Fast tests, no checkpoint and no GPU (0.2 s):

    cd /home/morg/NLP_2526b/galbarak2/LMEnt-mlp
    python -m unittest discover -s mlp_erasure/tests -p 'test_*.py' -v

On the cluster, `killable` / `a6000` -- the same target the EMBER erasure of
this concept used, chosen there to stay clear of the h100 pool:

    cd mlp_erasure
    sbatch --job-name=rmuprobe --export=ALL,MODE=probe,OUT_TAG=rome run_rmu.slurm
    sbatch --job-name=rmurun --export=ALL,MODE=run,STEERING=<from probe>,OUT_TAG=rome run_rmu.slurm

    sbatch --job-name=snmffact --export=ALL,MODE=factorize,OUT_TAG=rome run_snmf.slurm
    sbatch --job-name=judgesmk --export=ALL,MODE=judge-smoke,OUT_TAG=judge run_snmf.slurm
    sbatch --job-name=snmfsel --export=ALL,MODE=select,OUT_TAG=rome,SNMF_OUT=<dir> run_snmf.slurm
    sbatch --job-name=snmfver --export=ALL,MODE=verify,OUT_TAG=rome,SNMF_OUT=<dir> run_snmf.slurm
    sbatch --job-name=snmferase --export=ALL,MODE=erase,OUT_TAG=rome,SNMF_OUT=<dir>,SAVE_TO=<path> run_snmf.slurm

`MODE=erase` now defaults to `--delta-in 1 --delta-out 1`. If `run_snmf.slurm`
passes `--delta-in 4 --delta-out 4` explicitly, that line must go -- the job
will otherwise exit 1 at the guard rather than silently writing an amplified
checkpoint.

Everything lands under `/home/dcor/galbarak2/runs/mlp_erasure/`. `MODE=run`
does not keep the 4.4 GB model unless you export `SAVE_MODEL=` empty.

## What this does NOT establish

**No erased model has been produced or evaluated.** These scripts run; that is
all. Nothing here says RMU or SNMF removes Ancient Rome from the 1B, and
nothing here is comparable to `ERASURE_RESULTS.md`'s EMBER numbers yet.

**Both erasure hyperparameters now have a written rule, fixed ahead of the
runs.** RMU's `--steering` took the published grid value closest to 1x the
norm `MODE=probe` measured on this model, which gave 100 of {30, 100, 300,
1000}; SNMF's `tau` took the published 2.0. Neither was chosen after seeing an
effect size, and both rules were committed before the job that used them.
`ERASURE_RESULTS.md` is explicit that picking an erasure hyperparameter by the
size of the effect being measured is the failure that retracted two `acc_raw`
claims. What remains genuinely open is the step count -- see the limitation
section -- and that is a resourcing decision, not a tuning knob.

**The judge is a substitution, not the published one.** `--judge gemma` is
local gemma-4-12B-it in place of Gemini. It is the same judge EMBER's
embedding features went through here, which makes the two methods mutually
comparable, and neither of them comparable to the paper on that axis.

**One concept, one pair, one seed.** Everything in `ROME_RESULTS.md`'s scope
section applies unchanged.

---

# SUPERSEDED / UPDATED 2026-09-12

Two sections above are now out of date. Read this before acting on them.

## "What is now open, and it is not a code question" is CLOSED

That section frames `delta` and the verify metric as mutually inconsistent, with
the choice left to Gal. **It is not a choice.** `FEATURE_QUALITY_FIXES.md`
section 3 shows `configs/snmf_gemma.yaml` crosses `in_deltas`/`out_deltas`
`[1,4,7,10]` with every layer range and selects a cell on **downstream
evaluation** -- concept QA accuracy below 0.6, MMLU above 0.7 -- never on an
activation drop. The `.abs()` drop was never the selection criterion, so there
was never a contradiction to resolve. delta 4 amplifying the component threefold
is expected and irrelevant to how a cell is picked.

Anyone who read this file before 2026-09-12 and came away thinking a scientific
decision was pending should drop that belief. (It was repeated as a live open
item in conversation as recently as this session.)

## Verification runs, 2026-09-12: the suite and factorize, on 97443fe

Tamar's fixes landed on `main` as `97443fe`. Her file ends with an explicit
"do not push this without running, in this order" sequence. It was run:

**1. Unit suite -- PASSES.** 53 tests, 2.7 s, exit 0, under the real lment env.
That clears the tensor-semantics risk her doc lists as unverified (numpy shim
versus torch on `topk` tie-breaking, dtype promotion, and the
`arange().unsqueeze().expand()` sentence-id construction in
`collect_activations`, which had never executed).

**2. `MODE=factorize`, layers 4 5 6, job 883188 -- COMPLETED on an L40S.**

| layer | candidates rho>2.0, job 871493 | **candidates now, 883188** | max rho | scale inflation |
|---|---|---|---|---|
| 4 | 9 | **10** | 5.0118 | 1.07x |
| 5 | 34 | **31** | 4.7022 | 1.46x |
| 6 | 13 | **3** | 4.4545 | 1.23x |
| **total** | **56** | **44** | | |

Layers 4 and 5 are essentially unchanged; **layer 6 collapsed 13 -> 3**.

Two of her explicitly-unverified claims are now settled:

* **The semi-NMF convergence question is answered.** She wrote that whether
  `rtol=1e-6` converges on a real 5632x17000 matrix in a sane number of
  iterations "is an empirical question and nobody has answered it". It stops at
  **1052 / 1362 / 834 iterations** by patience, against `max_iter` 20000. The old
  absolute `tol=1e-4` against a reconstruction error of 1.43e10 could never fire.
* **Scale inflation is now visible and modest** (1.07x / 1.46x / 1.23x, with
  normalized medians 0.95-0.99), which is fix #2 working as intended.

**3. `MODE=select` -- the measurement, and it is NOT the factorize run.** Her doc
says the factorize job "is the one that answers the actual question ... the
accept rate against 871547's 10.7%". It is not: `factorize` produces candidates,
and the accept rate is produced by the judge in `MODE=select`. 44 candidates is
raw material, not an outcome. `select` runs against
`runs/mlp_erasure/snmf_rome_883188` and is gated behind `MODE=judge-smoke`
(job 883187), which is what checks the judge honours the new prompts and emits
TRASH.

**Still true and unchanged: no SNMF- or RMU-erased model has ever been built or
compared against the twins.** A higher accept rate would mean the judge agrees
more, which is not the same as the features being better. The honest test is
downstream -- erase, then measure against `M_never` the way
`ember_eval/ERASURE_RESULTS.md` does for EMBER.

## A trap in the drivers, fixed on a branch

`run_snmf.slurm` and `run_rmu.slurm` hard-assigned
`ROOT=/home/morg/NLP_2526b/galbarak2/LMEnt-mlp` -- not read from the environment,
not following `LMENT_ROOT`. A submission from any other checkout silently ran the
code in `LMEnt-mlp` and said nothing. Since `LMEnt-mlp` sits on
`feature/mlp-erasure` at `fa16e08`, which does **not** contain `97443fe`, running
the verification sequence from a checkout of `main` would have factorized with
the OLD `snmf.py` and reported a meaningless accept rate that looked entirely
normal. Every driver here now resolves `ROOT` by walking up from the directory
you submitted from, so a checkout runs its own code; `export ROOT=` still pins a
specific tree. Same class as
`activate_env.sh`'s `LMENT_ROOT` default.
