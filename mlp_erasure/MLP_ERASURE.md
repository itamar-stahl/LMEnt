# RMU and SNMF: making the two MLP erasure methods run on the LMEnt 1B

**Both scripts now run end to end on this project's models.** Neither had been
run here before. `snmf.py select` could not run at all -- it needed a Gemini
key this cluster does not have -- and both scripts loaded weights in a way that
would have made the resulting "erased" models unusable for the comparison they
exist to feed.

Branch `feature/mlp-erasure`, worktree
`/home/morg/NLP_2526b/galbarak2/LMEnt-mlp`, off `itamars/Ember-on-LMEnt`.

## Which two scripts, and why not the other pair

"The MLP erasure methods" here are `rmu.py` and `snmf.py` **at the repo root**,
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
  for the LMEnt 1B, and producing one is the job the root `snmf.py factorize`
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

Everything lands under `/home/dcor/galbarak2/runs/mlp_erasure/`. `MODE=run`
does not keep the 4.4 GB model unless you export `SAVE_MODEL=` empty.

## What this does NOT establish

**No erased model has been produced or evaluated.** These scripts run; that is
all. Nothing here says RMU or SNMF removes Ancient Rome from the 1B, and
nothing here is comparable to `ERASURE_RESULTS.md`'s EMBER numbers yet.

**Two hyperparameters are still unchosen, and both must be chosen before the
comparison, not after seeing it.** `--steering` for RMU (the published grid
{30, 100, 300, 1000} was tuned on wider models; `MODE=probe` reports the scale
this model actually produces) and `tau` for SNMF (`rho_stats.json` reports the
measured distribution). `ERASURE_RESULTS.md` is explicit that picking an
erasure hyperparameter by the size of the effect being measured is the failure
that retracted two `acc_raw` claims -- write the rule down first.

**The judge is a substitution, not the published one.** `--judge gemma` is
local gemma-4-12B-it in place of Gemini. It is the same judge EMBER's
embedding features went through here, which makes the two methods mutually
comparable, and neither of them comparable to the paper on that axis.

**One concept, one pair, one seed.** Everything in `ROME_RESULTS.md`'s scope
section applies unchanged.
