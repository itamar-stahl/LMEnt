# Why SNMF's features were weak, and what changed

Written against `snmf.py` and `rmu.py` at the repo root, checked line by line
against three references already in this tree:

- `Ember-on-LMEnt/ember/erasure/mlp_edit.py`, `features.py`, `methods/snmf.py`,
  `methods/rmu.py`, `config.py`, `configs/snmf_gemma.yaml` — the fork's own
  implementation and its published hyperparameters
- `Ember-on-LMEnt/external/snmf/` — the upstream paper code, in particular
  `factorization/seminmf.py` and `experiments/snmf_interp/`

The arithmetic claims below were run, not reasoned about. No GPU and no
checkpoint were involved, so nothing here supersedes a real run.

---

## 1. The bug: feature evidence was token positions, not token types

`top_tokens_per_feature` took the top-N columns of `Y` by coefficient. `Y` has
one column per token **position** in the corpus — about 17,000 of them in job
871247 — not one per token type. A feature that responds to a concept responds
at every occurrence of the tokens it likes, so the top N positions are the
same handful of strings repeated.

Simulated on a Zipf corpus the same size as the real one:

| feature supported by | distinct strings in the 20 slots sent to STAGE1 |
|---|---|
| 3 token types | **1** |
| 8 token types | 4 |
| 25 token types | 6 |

STAGE1 was being asked to name a concept from `['▁Rome','▁Rome','▁Rome', …]`.

It is worse than that, because the upstream reference does not pass bare
tokens at all:

- `generate_concept_context.py` emits `{token, context, activation}` with a
  ±15-token window **clipped to the sentence**, 25 examples per factor
- `generate_input_descriptions.py` renders them as
  ``Token: `t`, Context: `c` | Score: `s` `` and sends the top 10
- its prompt branches on exactly the duplicate case: *if the high-importance
  samples are mostly identical tokens, read the tokens; otherwise read the
  surrounding contexts.* That branch is dead without contexts.
- both upstream prompts have a **TRASH** output for a feature with no coherent
  connection. The old `STAGE1` had none, so the judge produced a description
  for every feature and STAGE2 then ruled on a confabulation.

This is sufficient on its own to explain a 10.2% accept rate and layer 4
returning zero twice.

### What changed

- `collect_activations` now also returns a per-token sentence id, so a context
  window can stop at the sentence boundary.
- `top_contexts_per_feature` replaces the bare-token path: `{token, context,
  activation}` per example, deduplicated with `--max-per-token-type` (default
  3) so a repetitive feature still reads as repetitive without crowding out
  every other token.
- `projection_tokens_per_feature` now carries the logit score with each token,
  which STAGE1_PROJECTION's first instruction needs.
- Two source-specific STAGE1 prompts ported from upstream, both with TRASH.
- `_judge_evidence` routes by source, parses the `Results:` section, and
  short-circuits TRASH without a STAGE2 call.
- Defaults moved to upstream's: `--top-tokens 25`, `--context-window 15`,
  `--judge-top-tokens 10`.
- An old `out/` directory still judges, with a loud warning naming how many
  features were read from bare strings.

---

## 2. The layer-14 rho inflation is a scale artifact, and it is now visible

`MLP_ERASURE.md` records median rho 1.05 / 2.21 / 3.05 at layers 4 / 9 / 14
and 63 of 100 features clearing tau=2.0 at layer 14, and proposes a
null-concept refit to test whether the confound is topical homogeneity.

You can settle the shape of it without a GPU. Take a random `Y`, multiply
**only the concept columns** by a constant, change nothing else:

| concept-side gain | median rho | n over tau=2.0 |
|---|---|---|
| 1.0× | 1.00 | 0/100 |
| 2.0× | 2.00 | 50/100 |
| 3.0× | 3.00 | 100/100 |

That is layer 14's row reproduced from a pure global magnitude gap with zero
per-feature signal. Eq. 3 is a ratio of mean coefficients, so anything raising
overall activation magnitude on one side lifts every feature equally.

### What changed

`mass_ratio_normalized` divides each token column by its L1 mass before taking
the ratio. `rho_stats.json` now records both distributions plus
`scale_inflation` = median(rho) / median(rho_norm), and `factorize` warns above
1.5×. A feature with rho 3 and rho_norm 1 is not a concept feature.

This does not replace the null-concept refit — it tells you in advance what
that refit will find, and it makes the diagnosis per-layer and per-feature
rather than per-run.

---

## 3. delta 4 is not "the published value", and the metric conflict dissolves

`MLP_ERASURE.md` frames the delta/verify inconsistency as an open scientific
choice. It is not open. `configs/snmf_gemma.yaml`:

```yaml
snmf:
  in_deltas:  [1.0, 4.0, 7.0, 10.0]
  out_deltas: [1.0, 4.0, 7.0, 10.0]
eval:
  min_mmlu: 0.7
  max_qa_acc: 0.6
```

`SNMFMethod.enumerate_hps` crosses every delta with every layer range and the
reference selects a cell on **downstream evaluation** — concept QA accuracy
under 0.6, MMLU over 0.7 — never on an activation drop. So the `.abs()` drop
was never the selection criterion and there is nothing to reconcile.

Confirmed numerically against `mlp_edit.intervene`:

| delta | component after/before | verify reports |
|---|---|---|
| 0.5 | 0.50 | +50% |
| 1 | 0.00 | +100% |
| 2 | 1.00 | 0% |
| 4 | 3.00 | −200% |
| 7 | 6.00 | −500% |
| 10 | 9.00 | −800% |

The comment block in `cmd_verify` now says this, and the advice no longer
implies there is a contradiction to resolve.

---

## 4. `verify` could not see the output side at all

`collect_activations` hooks the **input** to `down_proj`. Editing `down_proj`
does not change its own input — only what the layer writes to the residual
stream, which shows up downstream if anywhere. Every number in the old verify
table was reporting the `up_proj` edit alone, including at layers where
`--delta-out` was the only thing applied.

`feature_readout` added: mean `|W_out · f|` over the feature's support, a
weight statistic needing no forward pass. `verify` now prints both tables,
labelled, with the blind spot stated in the header.

---

## 5. The semi-NMF stopping rule was scale-dependent

`tol=1e-4` absolute, against a reconstruction error of 1.43e10 (layer 4) and
1.38e12 (layer 14). At 1.4e12, 1e-4 is 7e-17 of the loss and smaller than one
float32 step there (~8e4), so "improved" fired on any decrease at all and
patience effectively never triggered. Layers then stopped at whatever
iteration cap they hit, which is why layer 9's 55/100 could not be quoted
beside layers 4 and 14 — the three were fit to different degrees of
convergence because the criterion scaled with activation magnitude rather than
with progress.

`semi_nmf` now takes `rtol` (default 1e-6, relative to the running best) and
returns a third value, `info`, recording `stop_reason`, `converged`,
`best_iter` and `recon`. That lands in `rho_stats.json` per layer, and
`factorize` names any layer that hit the cap. `--ridge` default moved from
1e-4 to **1e-6**, which is the reference `reg` in `seminmf.py:fit`.

**Callers must unpack three values now.**

---

## 6. Only one of three published layer-range cells was reachable

```python
GEMMA_LAYER_RANGES_IN  = [(0, 25), (0, 8),  (0, 12)]
GEMMA_LAYER_RANGES_OUT = [(0, 8),  (9, 17), (13, 25)]
```

Cell 0 was treated as "the published default". This is what blocked job
871388: the judge's 13 features sat at layers 9 and 14, cell 0's output range
on 18 layers is [0,5], and the one-sided guard correctly refused.

`default_layer_ranges(n, cell)` and `--range-cell {0,1,2}` added. Verified
that all three cells reproduce the Gemma-2-2B indices exactly. On 18 layers:

| cell | in | out |
|---|---|---|
| 0 | (0,17) | (0,5) |
| 1 | (0,5) | (6,11) |
| 2 | (0,8) | **(9,17)** |

Cell 2 covers layers 9 and 14. Reaching it is not a departure from the
published grid — it is the rest of the published grid. The erasure is
unblocked without re-factorizing and without widening a range by hand.

---

## RMU

The port is faithful to WMDP on the things that matter: `down_proj` by name
rather than positional `param_ids=[6]` (correct for OLMo-2, where the index is
8), single-device load, frozen/updated split, pad masking. Three real
problems:

**Batch size.** `RMUGridConfig.batch_size` is **4** in the reference.
`rmu.py` defaulted to 16. That is the whole of the "19 steps instead of 150"
limitation the notes describe as a resourcing decision: 300 sentences at 16 is
19 batches, at 4 it is **75**. Same data, four times the optimiser steps, no
extra cost. Default changed to 4.

**`min_len` was dead.** `RMUGridConfig.min_len = 50` exists and is passed into
`_build_rmu_data`, which never applies it — the filter is in the config and
nowhere in the code. `build_batches` now applies both bounds and reports what
it dropped. `--min-len` / `--max-len` exposed.

**Cosine diagnostics lagged one step.** `cos_forget` / `cos_retain` were
computed from `f_act` / `r_act`, which were produced by the **pre-step**
weights, so the final update never appeared in `cos_forget_end` — the number
`forget_rotated` is judged on. Both are now re-read after `opt.step()`.

Also documented: `lr`, `alpha` and `steering` are grids upstream
(`[1e-5,1e-4,3e-4]`, `[30,50,100,300]`, `[30,100,300,1000]`) selected on
downstream eval. Norm-matching is a defensible way to pick one cell when you
can only afford one, but a single-cell run is not comparable to a swept
result, and `run_rmu`'s docstring now says so.

---

## What was verified, and how

`mlp_erasure/tests/test_feature_evidence.py` — 25 assertions across the
evidence pipeline, TRASH handling, rho scale control, relative convergence,
the range cells and the output-side readout. **Written but not executed
here:** no torch wheel is reachable from the sandbox this was done in.

> **Superseded — see "Executed against real torch" at the bottom of this
> file.** The suite has since been run against torch 2.13 on CPU. It passes,
> but running it also turned up a bug in `feature_readout` that the shim could
> not have caught, because the test written for it never called it.

What *was* executed: the same 25 checks against the patched `snmf.py` under a
numpy stand-in for torch, covering every function that does not touch a model,
autograd or dtypes. All 25 pass. The four arithmetic results quoted above
(delta scaling, duplicate collapse, rho inflation, float32 resolution) were
computed, not asserted.

**Run this before trusting any of it:**

```
python -m unittest discover -s mlp_erasure/tests -p 'test_*.py' -v
```

The existing `test_mlp_erasure.py` should still pass unchanged — the new
arguments all default, and `default_layer_ranges(26)` and `rho_summary(rho,
tau)` keep their old behaviour.

---

## Suggested order of work

1. Run the test suite. Fix whatever the shim could not see.
2. Re-factorize layers 4/5/6 with the new evidence capture. This is the run
   that matters — the same candidates will be judged on contexts instead of
   repeated strings, and the accept rate is the measurement.
3. Read `scale_inflation` in `rho_stats.json` before reading `n_gt_tau`.
4. Erase with `--range-cell 2` if the selected features sit deep, or stay on
   cell 0 if 4/5/6 produce enough.
5. Sweep delta over {1, 4, 7, 10} and select on the eval, the way the
   reference does. `verify` confirms the edit landed; it does not choose delta.
6. RMU: re-run at `--batch-size 4` for 75 steps before concluding anything
   about whether 19 was too few.

---

## Pre-push audit

Run after the changes above, before anything was proposed for commit.

### Found and fixed during the audit

**`run_rmu.slurm` pinned `BATCH_SIZE:=16`.** Changing the default in `rmu.py`
to 4 would have had no effect on any cluster run, because the slurm script
exports 16 over it. This was the single most consequential thing the audit
caught: without it the "19 optimiser steps" ceiling survives the fix that was
supposed to remove it. Now `:= 4`, with `MIN_LEN:=50` alongside it, and both
passed through to `rmu.py`.

**`judge_gpu_smoke.py` sent bare token lists.** It still ran after the
changes, but through the degraded no-context path — so the one test that
exists to prove the judge seam would have been exercising the format the fix
replaces. Rewritten to send both real formats (activation dicts with contexts,
projection dicts with scores) across four cases, and it now fails if
`has_context` is false or if the judge returns TRASH for a feature that
plainly is the concept.

**New flags were unreachable from slurm.** `--range-cell`, `--top-tokens`,
`--context-window`, `--max-per-token-type`, `--rtol` and `--judge-top-tokens`
are now threaded through `run_snmf.slurm` with defaults and comments.

### Checked clean

- `test_mlp_erasure.py` calls only `default_layer_ranges`, `resolve_ranges`,
  `rho_summary`, `ablate_layer`, `check_both_sides_applied`, `load_model`,
  `mlp_of`, `get_layers` and four `rmu` helpers. Every one keeps its old
  signature and behaviour; the new parameters all default. It should pass
  unchanged.
- No other file in the repo imports `snmf` or `rmu`. The `wmdp` and
  `Ember-on-LMEnt` copies are independent.
- Every flag both slurm scripts pass is declared in the corresponding
  argparse (two apparent misses were a comment and `BooleanOptionalAction`).
- Both slurm scripts pass `bash -n`.
- All five Python files parse; a hand-rolled ast name-resolution pass over
  them reports no undefined names (three hits are closure and `__file__`
  false positives).

### What was actually executed

| suite | what it covers | result |
|---|---|---|
| logic | evidence dedup and contexts, rendering, `Results:` parsing, TRASH, rho scale control, range cells, readout | 25/25 |
| command drivers | `cmd_factorize` end to end (both converged and capped), `cmd_select` against a scripted judge including a TRASH-only judge, `cmd_erase` through all three guards and the save path | 52/52 |

Both run `snmf.py` **as patched**, under a numpy stand-in for torch, with
`load_model` / `collect_activations` / `semi_nmf` /
`projection_tokens_per_feature` / `ablate_layer` faked. So the control flow,
tuple unpacking, print branches, guard conditions, JSON writes and metadata
of the real commands ran. Confirmed by observation: the scale-inflation
warning fires at 2.9–3.1×, the unconverged-layer warning names layer 5, the
one-sided guard refuses, and `--delta-out 0` is still allowed.

### What is still unverified, and it matters

**No torch, no model, no GPU touched this.** Not reachable from the
environment the work was done in. That leaves four things unchecked:

1. **Tensor semantics.** The numpy stand-in implements `topk`, `clamp`,
   `mean(dim=)`, boolean indexing and broadcasting the way numpy does. Where
   torch differs — dtype promotion, `topk` tie-breaking, `clamp` on integer
   tensors — the shim will not have caught it. The likeliest place for a real
   failure is `collect_activations`, whose sentence-id construction
   (`arange(...).unsqueeze(1).expand(...)`) never ran.
2. **`semi_nmf` with the relative tolerance.** The convergence change is
   argued from the arithmetic and the fit-info plumbing is tested, but the
   loop itself was stubbed in every run above. Whether `rtol=1e-6` converges
   on a real 5632×17000 activation matrix in a sane number of iterations is
   an empirical question and nobody has answered it.
3. **`feature_readout`.** New function, never executed.
4. **The judge.** The prompts are ported correctly, but whether
   gemma-4-12B-it follows the upstream Analysis/Results format and emits
   TRASH when it should is a fact about the model. `judge_gpu_smoke.py` is
   the test; it needs a GPU.

**Do not push this without running, in this order:**

```
python -m unittest discover -s mlp_erasure/tests -p 'test_*.py' -v
sbatch --export=ALL,MODE=judge-smoke,OUT_TAG=judge run_snmf.slurm
sbatch --export=ALL,MODE=factorize,OUT_TAG=rome,LAYERS='4 5 6' run_snmf.slurm
```

The first is fast and catches the tensor-semantics risk. The second checks
the judge against the new prompts. The third is the one that answers the
actual question — the same layers as job 871493, the same candidates, judged
on contexts instead of repeated strings, so the accept rate against 871547's
10.7% is the measurement of whether any of this worked.

---

## Executed against real torch

Everything above this section was written without torch. This section is the
follow-up run: torch 2.13.0, CPU, real `transformers`, no checkpoint and no
GPU. It supersedes the "what was verified" and "what is still unverified"
tables above wherever the two disagree.

### The suite passes

```
python -m unittest discover -s mlp_erasure/tests -p 'test_*.py' -v
Ran 53 tests ... OK
```

53, not 49, because the output-side class was rewritten — see below. The
tensor-semantics risk flagged as item 1 is clear: nothing in the numpy shim's
blind spot (dtype promotion, `topk` tie-breaking, `clamp`, broadcasting) turned
out to matter.

### `collect_activations` sentence ids are correct

The `arange(...).unsqueeze(1).expand(...)` construction named as the likeliest
real failure was run against a 6-layer Olmo2 with deliberately ragged
sentences, so padding was genuinely exercised. One unique id per sentence,
correct run lengths, non-decreasing, no pad token surviving the keep mask, and
context windows that stop at the sentence boundary on data that came out of
the real function rather than out of a fixture.

### `semi_nmf`'s stopping rule is scale-free, and demonstrably so

Same matrix at three magnitudes:

| scale | recon | stopped_at | best_iter | reason |
|---|---|---|---|---|
| 1e0 | 4.1481e+04 | 61 | 11 | patience |
| 1e3 | 4.1481e+10 | 61 | 11 | patience |
| 1e6 | 4.1481e+16 | 61 | 11 | patience |

Identical. The old absolute rule could not do this: at a reconstruction error
of 1.33e16 the float32 spacing is ~7.95e8, so `tol=1e-4` sat thirteen orders of
magnitude below one representable step. `rtol=1e-6` terminated on patience at
(200,800,k=30) in 2560 iterations and at (512,2000,k=50) in 684.

**Still open:** a (1024,4000,k=100) fit had not converged after 25 minutes of
CPU, so how many iterations `rtol=1e-6` needs on a real 5632x17000 matrix is
still unanswered. It is no longer silent, though — `info["stop_reason"]` says
which way the layer stopped and `factorize` names any layer that hit the cap.

### The evidence reaching STAGE1 is genuinely fixed

On a Zipf corpus of the same shape as job 871247 (17,100 positions, 300
sentences, 2,677 distinct strings):

| feature supported by | OLD distinct / 20 | NEW distinct | NEW items | with context |
|---|---|---|---|---|
| 3 types | 3 | 17 | 25 | 25 |
| 8 types | 6 | 9 | 25 | 25 |
| 25 types | 12 | 15 | 25 | 25 |
| 60 types | 12 | 19 | 25 | 25 |

Every example carries a context, every context is contained in a single
sentence (checked across all 25, not on a fixture), and the window respects
+/-15 tokens. The old path saturates around 12 distinct strings in 20 slots
regardless of how broad the feature is.

### Bug found by running it: `feature_readout` measured noise at delta=1

`test_readout_falls_when_down_proj_is_projected` re-implemented the projection
inline and **never called `snmf.feature_readout`**, so the function shipped
untested while appearing in the table above as covered.

Called for real, after an exact `delta_out=1.0` edit:

| | before | after | reported |
|---|---|---|---|
| `feature_readout`, fp32 | 0.03306 | 0.00883 | 73% drop |
| `feature_readout`, fp64 | 0.03306 | 0.00000 | 100% drop |
| against a fixed direction | 0.03306 | 4.97e-9 | 100% drop |

Cause: it recomputed `f = W_out^T z` from the **edited** weights, where that
quantity is analytically zero — `f_new = u - delta*(u.f_hat) f_hat`, which at
delta=1 is `u - u`. The edit lands exactly as intended (`||W_out^T z||`
collapses 2.58e-1 -> 3.92e-8); the function then divided that cancellation
residue by its own norm, turning rounding error into a unit vector. In float64
the residue falls under the `clamp(min=1e-8)` instead, so the same call returns
a different answer. At delta != 1 the recomputed direction stays parallel to
the original and `.abs()` hides the sign, which is why only delta=1 was wrong —
and delta=1 is both the exact-removal case and one of the four values in the
published sweep {1, 4, 7, 10}.

**Fixed** by adding `feature_directions()`, which captures the unit readout
directions from the unedited weights, and giving `feature_readout` an optional
`directions` argument. `cmd_verify` now takes the directions once before the
edit and reuses them after. With the fix, the readout reproduces the law in
`cmd_verify`'s own comment block exactly, in both dtypes:

| delta | 0.5 | 1 | 2 | 4 | 7 | 10 |
|---|---|---|---|---|---|---|
| reported drop | 50% | 100% | 0% | -200% | -500% | -800% |
| predicted `1-abs(1-delta)` | 50% | 100% | 0% | -200% | -500% | -800% |

The test class was rewritten to call the real function, to cover the
empty-support branch and the delta=2 sign flip, to assert the captured
directions are not a view into the weights, and to pin the old broken
behaviour so the fix cannot be quietly reverted.

### Still not verified here

The judge. `gemma-4-12B-it` following the upstream Analysis/Results format and
emitting TRASH when it should is a fact about the model, and
`judge_gpu_smoke.py` needs a GPU. Nothing in this section says anything about
which features a real run selects — the accept rate against 871547's 10.7% is
still the measurement, and it still requires the checkpoint.
