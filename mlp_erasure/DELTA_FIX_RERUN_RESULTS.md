# MLP erasure re-run on the fixed code (2026-09-14)

Everything in `ERASURE_RESULTS.md` and the earlier MLP null tables was produced
before the delta and retain-cycling bugs in `fix/erasure-delta-and-rmu-steps`
were found. This file records the re-run on the fixed code, plus the controls
that were missing the first time.

Scoring is `ember_eval/score_ember_mc.py` (`text_char`) or `sweep_delta.py`'s
`causal_mc` scorer; the two were cross-validated against each other and agree
within a point (see "Full layer coverage" below). n = 50 per split, chance 25%,
so the standard error is roughly +-7 points: **2 points is one question.**

Model shorthand: `control` = `lment-1b-control-2e-b131k`.

---

## 1. The bug that invalidated the old table

`ablate_layer` applies `(I - delta*P)` onto a **unit** direction, so the
targeted component scales by `(1 - delta)` — take the magnitude and that is
`|1 - delta|`:

| delta | component becomes | |
|---|---|---|
| 0.5 | x +0.5 | halved, sign kept |
| **1.0** | **x 0** | **removed exactly — the only true erasure** |
| 2.0 | x -1 | same size, sign flipped |
| 4.0 | x -3 | tripled, sign flipped |
| 10.0 | x -9 | 9x, sign flipped |
| -0.5 | x +1.5 | strengthened, sign kept |
| -1.0 | x +2 | doubled, sign kept |
| -19.0 | x +20 | 20x, sign kept |

**Every delta in the original investigation was 4, 10 or 20** — all amplifying,
all sign-flipping. The erasure setting was never tested. That is what this
re-run fixes.

---

## 2. SNMF at the real erasure delta: null

Held-out test split, `score_ember_mc.py` `text_char`.

| feature set | model | control | delta=1 |
|---|---|---|---|
| Rome judge-selected (6 feat, L5-6) | control | 44.0% | **44.0%** |
| Rome ratio-only (44 feat, L4-6) | control | 44.0% | **48.0%** |
| Baseball ratio-only (58 feat, L4-6) | control | 48.0% | **46.0%** |
| AI ratio-only (84 feat, L4-6) | control | 44.0% | **46.0%** |
| Rome ratio-only (51 feat, L4-6) | Daniela LMEnt-1B-2E | 40.0% | **42.0%** |

All four checkpoints, all three concepts: nothing moves, and in three of five
rows concept accuracy goes slightly *up*.

### Artificial intelligence (the last cross-topic hole)

The only AI SNMF numbers previously on record were from the buggy delta=4 era
(-6pt, single seed, flagged in `lment-crosstopic-mlp-erasure.md` as probably
noise). Re-run on the fixed code with the largest feature set of the four
(84 features, L4:28 L5:45 L6:11):

| | concept | simdom |
|---|---|---|
| control (test) | 44.0% | 54.0% |
| delta=1, explicit layers 4-6 both sides | **46.0%** | 46.0% |
| delta=1, permutation controls | 44.0 / 44.0 / 44.0 (= control exactly) | |
| delta=-19 (x20), real | 36.0% | 38.0% |
| delta=-19, permutation controls | 38.0 / 38.0 / 46.0 | 44 / 42 / 40 |

Train grid, all three range cells at delta=1: 38.0 / 36.0 / 40.0 against a
40.0% control. At the amplifying deltas **selectivity is negative in every
single cell** (-4 to -18): simdom is damaged more than concept, and cell 2 at
delta=10 puts concept *up* 6 while simdom falls 12. The old -6pt is retired.

AI also tests the feature-count hypothesis from section 4 most sharply, having
nearly twice Rome's features. Its x20 concept drop (-8) is comparable to
Rome's (-6) while its collateral is much heavier (-16 simdom), so "more
features -> more damage" holds loosely for *total* damage but the damage is
non-specific either way.

> Operational note: the first AI job (895359) died mid-grid with
> `CUDA error: unspecified launch failure` on n-301, after which every process
> on that node silently fell back to CPU — the same silent-fallback shape as
> n-102's dead GPU. Stages 2-3 there are still valid (fresh model loads, and
> stage 3 reproduced the known control baseline 44.0/54.0 exactly), and the
> grid was re-run clean on n-302 (895384) with cell 0 reproducing the
> pre-fault numbers exactly. Jobs 893308 and 893739 also ran on n-301 and were
> checked for the same fault: zero occurrences, so those findings stand.

Train-split sweeps over all three published layer-range cells agree: flat
everywhere at delta=1.

### Full layer coverage — the RANGE_CELL loophole, closed

`sweep_delta.py` only exposes `--range-cells`, and cell 0's **output** range on
18 layers is `[0,5]`, so selected features at layer 6 had their output side
silently skipped (`cmd_erase` `continue`s past out-of-range layers without
erroring). Re-run with explicit `--layers-in/--layers-out` covering every layer
holding a selected feature, on both sides:

| cell | features ablated | QA_test | control | SimdomQA_test |
|---|---|---|---|---|
| Rome judge | 6/6, 2 layers both sides | 44.0% | 44.0% | 54.0% (52.0%) |
| Rome ratio44 | 44/44, 3 layers both sides | 48.0% | 44.0% | 54.0% (52.0%) |
| Baseball58 | 58/58, 3 layers both sides | 46.0% | 48.0% | 20.0% (24.0%) |

Full coverage changes nothing, and these land within a point of
`sweep_delta.py`'s independent scorer.

---

## 3. Two-sided causal null

A one-sided null (removal does nothing) is compatible with the model
compensating for the edit. A two-sided null is not. Train split, control:

| intervention | component | Rome judge (48%) | Rome ratio44 (48%) | Baseball58 (46%) |
|---|---|---|---|---|
| delta 0.5 | x0.5 | **48.0% exactly, all cells** | 50.0% | 44.0% |
| delta -0.5 | x1.5 | 46-48% | 44-48% | 46% |
| delta -1.0 | x2 | 46-48% | 44-46% | 46-48% |
| delta -2.0 | x3 | 46-48% | 44-46% | 44-46% |

Rome-judge at delta 0.5 reproduced the control numbers **exactly** on both
splits in all three cells. Scaling these directions from 0x to 3x
sign-preserved, or inverting them to -9x, does not move concept accuracy.

**Scope limit:** this shows these *particular SNMF-identified directions* are
causally inert for this QA task. It does not show MLPs are irrelevant — a
different feature-identification method could still find directions that matter.

---

## 4. Large amplification: capacity damage, not erasure

At x9/x20/x50/x100 (deltas -8/-19/-49/-99) accuracy finally moves — but the
damage tracks **feature count**, not concept:

| feature set | n | concept (control) | at x20-x100 |
|---|---|---|---|
| Rome judge | **6** | 48% | **40-52% — nothing** |
| Rome ratio | 44 | 48% | 32-38% |
| Baseball | 58 | 46% | 26-40% |
| Daniela | 51 | 46% | 28-42% |

Six features at x100 does nothing; 44-58 features degrades the model.
Daniela's model degrades non-selectively (simdom falls MORE than concept,
selectivity -2 to -22). Baseball cell 2 at x100 puts simdom at **14%, below
the 25% chance floor** — the model is simply broken there.

### The permutation control

`random_direction_control.py` shuffles each selected feature's entries across
neurons: **identical values, identical support size, identical magnitude
distribution, different neurons.** Validity confirmed by the `edited=` count
matching exactly (44/58/6) between real and every permuted seed.

| set / delta | real | perm mean | separation |
|---|---|---|---|
| Rome-44, x20 | 42.0% | 40.7% | none |
| Rome-44, x100 | 34.0% | 32.7% | none |
| Rome-6, x20 | 44.0% | 43.3% | none |
| Rome-6, x100 | 50.0% | 40.7% | none (real higher) |
| Baseball-58, x20 | **30.0%** | **44.7%** | large |
| Baseball-58, x100 | 32.0% | 38.0% | moderate |

For Rome the real features sit inside the permutation distribution: that
regime is capacity destruction.

---

## 5. The Baseball separation, and why it still does not count

Baseball seed-42 at x20 on **held-out test**: concept 48% -> **26.0%** (chance
25) with simdom *up* (+6), against permutation controls at 46/44/38%.
Consistent across train (30%) and test (26%) and across x20/x100. As an effect
of those 58 directions it is solid.

**But independent factorizations of the same concept show nothing:**

| arm | real | perm mean | separation |
|---|---|---|---|
| seed1 (47 feat), x20 | 38.0% | 41.3% | none (a perm hit 36%) |
| seed1, x100 | 40.0% | 40.7% | none |
| seed2 (50 feat), x20 | 38.0% | **38.0%** | exactly zero |
| seed2, x100 | 30.0% | 37.3% | marginal |
| **seed42, x20 (test)** | **26.0%** | **42.7%** | large |

seed1/seed2 are equally valid semi-NMF decompositions of Baseball on the same
model — only the random init differs. **So the effect is a property of the
seed-42 decomposition, not of Baseball's representation:** reproducible on
unseen questions, not reproducible under a different decomposition of the same
knowledge.

This is the **third** Baseball SNMF result to die on a seed check (the pre-fix
"-8pt at delta 4" gave 40/46/50% across seeds; then the cross-topic SNMF lead;
now this).

> **Standing rule: no SNMF result counts until it reproduces across independent
> factorization seeds, and a permutation-of-neurons null is the right control
> for any large-delta effect.**

---

## 6. RMU on the fixed code

Rome depth grid on the control, with retain-cycling fixed and scored against
the new `rotation_is_substantial` (>= 0.30) / `ran_enough_steps` (>= 50%) gates:

| cell | QA_test | SimdomQA_test | cos_forget | substantial | enough steps |
|---|---|---|---|---|---|
| L4 mid | 24.0% | 36.0% | 0.022->0.050 | no | no (47%) |
| L4 hi | 42.0% | 54.0% | 0.034->0.243 | no | no (47%) |
| L5 mid | 46.0% | 52.0% | 0.028->0.072 | no | no (47%) |
| **L5 hi** | **44.0%** | 50.0% | 0.049->**0.314** | **yes** | no (47%) |
| L6 mid | 46.0% | 52.0% | 0.018->0.061 | no | no (47%) |
| **L6 hi** | **44.0%** | 50.0% | 0.040->**0.319** | **yes** | no (47%) |

Control: 44.0% / 52.0%. **The only two cells that achieved a real misdirection
land exactly at control.** The other four never cleared the rotation gate, so
their numbers are not evidence either way — L4-mid's 24.0% in particular comes
from a cell that fails even the loose `forget_rotated` bar and is best read as
instability.

**Every cell fails `ran_enough_steps` at 47%:** the forget corpus (300 Rome
sentences -> 71 batches at batch-size 4) caps below the 50% bar regardless of
the retain-cycling fix, which only removed the *retain*-side cap. Closing that
needs more concept sentences (`Ember-on-LMEnt/sentences_gen`) or a smaller
batch.

### RMU on Daniela's model is non-functional, not null

| steering | 30x norm | 60x norm | 100x norm | L6 @ 100x |
|---|---|---|---|---|
| `cos_forget_end` | 0.035 | 0.036 | 0.036 | 0.041 |
| QA_test | 40.0% | 40.0% | 40.0% | 40.0% |

All four bit-identical to her baseline (40.0% / 62.0%). A 3.3x steering range
moved rotation by 0.001, and the unlearn loss fell 5574.9 -> 5569.8 over 71
steps (0.09%). RMU as configured (lr 1e-4, alpha 100, ~71 steps) **cannot
engage** on her checkpoint anywhere in the 10-100x band, while the same
relative steering reaches 0.31-0.37 on the control. **Do not report her flat
QA as an erasure null — it is an instrumentation failure.** Untested: much
lower steering (1-5x, comparable to her activation norms) and/or higher lr.

---

## 7. Cross-checkpoint replication (Daniela Gottesman's LMEnt-1B-2E)

Her released 2E checkpoint and the control are **token-matched** — 219,344
steps x batch 32,768 vs 54,832 x 131,072, both ~7.19B tokens — differing only
in the batch/step tradeoff, which makes this a clean replication.

SNMF null replicates (section 2). The one cross-model difference is
**fragility, not erasure**: her model degrades under large sign-flipped deltas
(cell 1 delta 10: concept 46->30, simdom 52->30) where the control barely
moves, but every such cell has selectivity <= 0. Her residual norms are ~2.4x
smaller (27.4/33.8/41.8 vs 72.7/83.2/93.8), a plausible mechanism.

**Answer to "is there a hyperparameter that erases on her model but not ours":
no.**

### Operational notes for her checkpoints

- The `dhgottesman/LMEnt-1B-*` repos are laid out **by checkpoint step
  subfolder** (`step219344` for 2E, `step109672` for 1E), not as flat model
  dirs. Pointing at the bare snapshot dir makes `AutoTokenizer` fail with a
  confusing ESM-tokenizer error.
- `ember/slurm_model.py` whitelists only the two control paths, so
  `prepare_submission` needs `validate_cluster_model=False` for any other model.

---

## 8. Post-EMBER composition

Does anything remain in the MLPs once EMBER has taken out the embedding-level
knowledge?

**The premise is verified, not assumed.** EMBER's configs run with
`save.full_model: false`, and hashing the tensors confirms it (job 893748
preflight):

    rome-erased:      DIFF embed_tokens | SAME mlp.down_proj | SAME mlp.up_proj
    baseball-erased:  DIFF embed_tokens | SAME mlp.down_proj | SAME mlp.up_proj

So an EMBER-erased checkpoint has **bit-identical MLP weights** to the
control; only what feeds them changed.

### Variant 1 — SNMF with control-derived features (job 893748), held-out test

The loaded-model baselines independently reproduce EMBER's own published
numbers: Rome-erased 38.0% (control 44.0%, i.e. -6) and Baseball-erased 36.0%
(control 48.0%, i.e. -12).

| model | EMBER baseline | delta=1 real | delta=1 perms |
|---|---|---|---|
| Rome-erased, judge 6 feat | 38.0% | **38.0%** | 38.0 / 38.0 / 38.0 |
| Rome-erased, ratio 44 feat | 38.0% | **38.0%** | 38.0 / 38.0 / 38.0 |
| Baseball-erased, 58 feat | 36.0% | **36.0%** | 36.0 / 36.0 / 36.0 |

**Not a single question flipped**, in any condition, on either split. There is
nothing left in the MLPs for these directions to remove.

At delta -19 (x20) there is no real-vs-permuted separation either:

| set | real | perms |
|---|---|---|
| Rome judge | 44.0% (concept *up* 6) | 36 / 34 / 40 |
| Rome ratio44 | 40.0% | 36 / 38 / 40 |
| Baseball58 | 32.0% | 34 / 38 / 34 |

**Baseball's seed-42 x20 effect disappears after EMBER** — 32.0% against perms
averaging 35.3%, where on the control it was 26.0% against 44.7%. Two readings,
and the data here cannot separate them: either the effect was mediated by
embedding-carried information that EMBER removed (which would make it real but
downstream of the embeddings), or it is a floor effect, since Baseball-erased
starts at 36% with only ~11 points of headroom above chance versus 23 on the
control. Worth stating as open rather than resolved.

### Variant 2 — SNMF re-derived ON the erased model (job 893749), held-out test

EMBER changed the embeddings, so the activations feeding the MLPs changed and
a fresh semi-NMF sees a different decomposition even though the weights are
untouched. It does find a different feature set — 40 features on Rome-erased
and 32 on Baseball-erased, against 44 and 58 derived from the control — so the
re-derivation is real, not a no-op.

It makes no difference:

| model | baseline | delta=1 real | delta=1 perms | delta=-19 real | delta=-19 perms |
|---|---|---|---|---|---|
| Rome-erased (40 feat) | 38.0% | 36.0% | 38.0 / 38.0 / 38.0 | 42.0% (*up 4*) | 38 / 36 / 32 |
| Baseball-erased (32 feat) | 36.0% | **36.0%** | 36.0 / 36.0 / 36.0 | 36.0% | 40 / 44 / 44 |

At delta=1 the real features move concept by at most one question, and the
permutations move it by zero. At x20 the real features land *above* every
permutation on Rome. **Re-deriving the decomposition after an embedding edit
does not recover any erasure ability** — that methodological worry is answered
in the negative.

### Variant 3 — RMU on the erased model (job 893750), held-out test

| cell | cos_forget | substantial | QA_test | baseline |
|---|---|---|---|---|
| Rome-erased **L5-hi** | 0.049 -> **0.302** | **yes** | **42.0%** | 38.0% |
| Rome-erased L6-hi | 0.037 -> 0.287 | no | 42.0% | 38.0% |
| Baseball-erased L5-hi | 0.043 -> 0.247 | no | 38.0% | 36.0% |
| Baseball-erased L6-hi | 0.034 -> 0.234 | no | 38.0% | 36.0% |

The one cell that achieved a genuine misdirection (Rome L5-hi, just clearing
0.30) left concept accuracy **4 points HIGHER** than the un-edited erased
model. Every cell again fails `ran_enough_steps` (71-73 of 150).

Note these rotations (0.23-0.30) are in the same range as the control's
0.31/0.32 and nothing like Daniela's 0.034 — as expected, since the
EMBER-erased checkpoints carry the control's own MLP weights and activation
scale. RMU engaged here; it simply did nothing.

### Composition verdict

Three methods — inherited SNMF directions, freshly re-derived SNMF directions,
and RMU — all applied after EMBER, all on held-out test, all null. At the true
erasure setting not a single question changes. **Once the embedding-level
knowledge is gone, there is nothing these MLP methods can reach.**

One open thread: Baseball's seed-42 x20 effect (section 5) vanishes on the
EMBER-erased model (32.0% vs perms averaging 35.3%, where on the control it
was 26.0% vs 44.7%). That is consistent either with the effect having been
mediated by embedding-carried information, or with a floor effect — the
erased model starts at 36% with ~11 points of headroom above chance versus 23
on the control. This data cannot separate the two.

---

## Bottom line

| method | level | result |
|---|---|---|
| EMBER | embedding | **works** — Rome -6, Baseball -12, AI -12, adjacent held |
| SNMF | MLP | two-sided causal null, 3 concepts, 2 checkpoints, 3 depth bands, **and on held-out chunk loss** |
| RMU | MLP | null at the full published schedule, both sanity gates green |

The reachable knowledge for these evals lives in the token embeddings, not in
the MLP directions these methods identify — consistent with EMBER's own thesis.

**Strength of the null, stated precisely.** It is not "we tried some settings
and nothing happened". For SNMF the directions can be scaled to 0x, 0.5x,
1.5x, 2x, 3x or inverted to -9x with no response, at three depth bands, on
three concepts, on two independently trained checkpoints, with neuron
permutation controls showing that whatever damage large edits do cause is not
specific to those directions — and on held-out chunk loss, millions of tokens
rather than 100 questions, both delta=1 models sit within 0.006 nats/token of
the control model while the same measurement puts EMBER at +1.32. For RMU the
method now runs at 100% of its
published schedule at the aggressive end of its retain penalty, achieving the
strongest representational rotation this project has produced (0.507), with
both sanity gates green — and concept accuracy goes UP. And after EMBER has
removed the embedding-level knowledge, neither method flips a single question.

The instrument qualifier is now discharged (GAP 3 below). The same erasures
re-measured on held-out chunk loss land within 0.006 nats/token of the control
model, on a probe that registers EMBER at +1.32 and never-having-learned at
+0.31. A more sensitive instrument sees nothing MC did not.

### Scope limits, and what was done about them

The claim above was initially scoped to *these methods as configured,
measured by this eval*. Three gaps kept it from being "MLP erasure does not
work on this model". **All three are now closed.**

#### GAP 1 — CLOSED. RMU had never run at its specified strength.

Every cell up to this point failed `ran_enough_steps` at 71/150. The cap was
arithmetic, not a choice: 283 usable sentences / batch 4 = 71 batches, and
`n = min(max_num_batches, forget)`. `alpha=100` compounded it — that is the
TOP of the reference grid, the heaviest retain penalty and so the least
unlearning available. So "RMU does nothing here" had meant "RMU at 47% of its
schedule, at its most conservative alpha, does nothing here".

At batch 1, 283 sentences give 283 batches, so `min(150, 283)` = **the full
published schedule**. Job 895391, L5-hi, steering 831.6:

| cell | batch | alpha | steps | cos_forget | both gates | QA_test | SimdomQA |
|---|---|---|---|---|---|---|---|
| b1_a100 | 1 | 100 | **150/150** | 0.063 -> 0.366 | **pass** | **50.0%** | 52.0% |
| **b1_a10** | 1 | 10 | **150/150** | 0.066 -> **0.507** | **pass** | **46.0%** | 50.0% |
| b2_a100 | 2 | 100 | 142/150 (95%) | 0.070 -> 0.418 | **pass** | 48.0% | 52.0% |
| b4_a10 | 4 | 10 | 71/150 (47%) | 0.050 -> 0.408 | rotation only | 48.0% | 52.0% |

Control: 44.0% / 52.0%. **Three cells pass BOTH sanity gates — the first time
that has ever happened in this project — and every cell lands at or ABOVE
control.** `b1_a10` is the maximum-unlearning configuration the reference grid
allows (full schedule *and* aggressive retain penalty); it produced
**cos_forget 0.507, the strongest representational rotation ever achieved
here** (previous best 0.494), and concept accuracy went *up* 2 points.

Rotation and downstream accuracy are not weakly related on this model. They
are unrelated. The RMU objection is closed: run as specified, at maximum
available strength, with the gates green, it does nothing.

#### GAP 2 — CLOSED. SNMF had only ever seen layers 4/5/6.

RMU got a depth battery out to layer 17; SNMF never left the 22-34% band. Job
895392 factorized fresh at two untouched bands, ratio-only selection, and
repeated delta=1 with permutation controls on held-out test:

| band | features | delta=1 real | delta=1 perms | delta=-19 real | delta=-19 perms |
|---|---|---|---|---|---|
| mid, L9-11 (~50-61% depth) | 154 | **44.0%** (= control exactly) | 46 / 44 / 46 | 38.0% | 36 / 42 / 34 |
| deep, L14-16 (~78-89% depth) | 125 | **48.0%** (*up* 4) | 44 / 44 / 44 | 36.0% | 36 / 46 / 40 |

Control 44.0%. Deeper factorizations select far MORE concept-selective
directions (154 and 125 against the shallow band's 44) and they are just as
inert: at delta=1 the mid band returns the control value exactly and the deep
band goes up, while at x20 real sits squarely inside the permutation spread in
both bands. **The SNMF null generalises across depth.**

#### GAP 3 — CLOSED. The sensitive instrument agrees.

Everything above is 4-option MC at n=50 (+-7pt). This project's own record says
chunk loss is the sensitive instrument — the Rome ablation showed far more
clearly there (+0.3144 nats/token on held-out Rome chunks) than on raw
accuracy. So the same erasures were re-measured on held-out chunk loss, with
**EMBER as the positive control** and a falsifier fixed in advance: if chunk
loss could not see EMBER's -6pt either, the instrument was not sensitive enough
here and the MC comparison should have been withdrawn rather than defended.

All four models were scored against **identical sets** — 16,915 held-out Rome
chunks (17,597,568 tokens) and 5,044 length-matched control chunks (5,221,206
tokens), seed 42, fp32. The control ids are cached to
`runs/mlp_erasure/rome_control_ids_seed42.json` and keyed on
blacklist/n_control/seed/dataset size, so a mismatched sample is refused rather
than silently scored.

| model | held-out | control set | **held-out − control** | vs control model |
|---|---|---|---|---|
| control (saw Rome) | 2.5075 | 2.3739 | **+0.1336** | — |
| snmf_judge_d1 (d=1, L5-6, 6 feats) | 2.5087 | 2.3742 | **+0.1344** | **+0.0008** |
| snmf_ratio44_d1 (d=1, L4-6, 44 feats) | 2.5190 | 2.3793 | **+0.1397** | **+0.0061** |
| ablated twin (never saw Rome) | — | — | **+0.3144** | +0.1808 |
| EMBER-erased | 3.7234 | 2.4021 | **+1.3214** | +1.1878 |

**The falsifier did not fire.** Chunk loss sees EMBER at +1.3214 — roughly ten
times the control model's gap and four times the ablated twin's — so the
instrument demonstrably has range on exactly this comparison. EMBER's own
control-set loss barely moves (2.3739 -> 2.4021, +0.028), so that is Rome-
specific damage, not general degradation.

**And both SNMF erasures land on top of the control model**: +0.0008 and
+0.0061 nats/token, against a probe registering +1.19 for EMBER and +0.18 for
never having learned the material. The MC null is not an artefact of a blunt
instrument. It replicates on the sensitive one.

Read the EMBER number as sensitivity, not as depth of erasure. It massively
overshoots the ablated twin, which is the ceiling for "this knowledge was never
acquired". That is the lexical signature this project has already characterised
— erasure damage is unbounded on chunks containing an edited token and zero
without one — and EMBER edits token embeddings, so every held-out Rome chunk
contains tokens it altered. It is the right positive control for whether the
probe can see anything; it is not a target the MLP methods should be expected
to approach.

Four tooling traps hit on the way, all worth knowing:

- **`ModuleNotFoundError: No module named 'olmo_core.data'`** (job 895390).
  The `lment` env ships a PARTIAL `olmo_core` with no `data/`, so the import
  fails in a way that reads like the package is absent. `framework/env.sh`
  puts the repo's complete sources on PYTHONPATH ahead of site-packages —
  and **env.sh itself defaults `LMENT_ROOT` to `<user root>/LMEnt`**, the same
  wrong-checkout trap as `activate_env.sh` and `set_node_env.sh`, so pin
  `LMENT_ROOT` before sourcing it.
- **`KeyError: 'chunk_ids'`** (job 895460). `heldout_ppl.py` reads
  `entities[0].chunk_ids`, i.e. it needs the RESOLVED blacklist that
  `prepare.py` materializes into each training run dir as
  `untaught_blacklist.json`, not the source QID list in
  `Untaught/blacklists/`. Pointing at the Rome twin's own copy is the right
  choice: the held-out chunks are then exactly the chunks held out of that
  twin's training, which is what makes the +0.3144 reference comparable.
- **Scoring off the share does not finish** (job 895592). `heldout_ppl.py`
  hardcoded its dataset paths to the morg filer, and it reads at RANDOM offsets
  — `match_by_length` draws up to `n_control * 40` instances and `chunk_losses`
  fetches every id it scores. Measured on the node: **287 read syscalls in 30s,
  313 KB/s**, process parked in `D` state on `rpc_wait_bit_killable` with the
  GPU at 0%. It spent two hours inside `match_by_length` for the FIRST of four
  models and would never have finished in its 4h wall. `DATA_GLOB`/`WORK_DIR`
  now follow `LMENT_DATASET`, so the runner sources
  `framework/node/stage_dataset.sh` and reads node-local exactly as training
  does; the same sampling pass then took **8.7 minutes**. Staging changes where
  the bytes come from, never which bytes or in what order.
- **`CUDA error: unspecified launch failure`** (job 897155, node n-307). A card
  fault, not a code fault: it hit after cleanly scoring all 16,915 held-out
  chunks of the second model. Resubmitting the one missing cell elsewhere was
  enough. `--exclude` that node, and note that n-102 is worse — its dead card
  poisons NVML for the whole host, so jobs assigned a *healthy* GPU there still
  die at distributed init.

The first of those failures exited **0 with zero result files** — a "completed
run with an empty answer". The scorer now aborts loudly if any model produces
no output.

**Budget four models across separate jobs.** Scoring one model is 21,959
chunks and runs ~75 min on a 3090 even with the dataset local, so four models
never fit one 4h wall. The control ids are therefore cached to a
job-independent path and reused, which both saves the sampling pass and
guarantees every model is scored against the same control set — the split that
has faked an effect in this project before.

## Reproducing

Scripts are in this directory (`run_snmf_*.slurm`, `run_rmu_*.slurm`,
`run_post_ember_*.slurm`) and `../random_direction_control.py`. The GAP 3
numbers come from `run_heldout_chunkloss_staged.slurm` (control + EMBER, job
896121), `run_heldout_chunkloss_snmf.slurm` (`snmf_ratio44_d1`, job 897155) and
`run_heldout_chunkloss_judge.slurm` (`snmf_judge_d1`, job 898136). The original
`run_heldout_chunkloss.slurm` is superseded — it reads the dataset off the
share and tries all four models in one 4h job. They bypass
`run_snmf.slurm`/`run_rmu.slurm`, which hardcode
`ROOT=/home/morg/.../LMEnt-mlp` and would silently execute the OLD, unfixed
code. Raw CSV/JSON outputs are gitignored; they live in
`/home/dcor/galbarak2/runs/mlp_erasure/` and are mirrored to
`/home/morg/NLP_2526b/galbarak2/backups/mlp-erasure-rerun-2026-09-14/`.
