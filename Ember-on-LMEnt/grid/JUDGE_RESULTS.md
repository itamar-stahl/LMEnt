# Stage 3: what the Gemma judge said about the 27 cells

Written 2026-09-08. Supersedes the negative result in `README.md`.

**Headline: EMBER does find an Ancient Rome feature in this 1B model.** Nine of
the 24 cells judged so far carry at least one feature the judge calls Ancient
Rome above the 0.85 confidence threshold. The single failing run that motivated
this whole grid (job 858233) was a bad draw, not a wall.

Judge `google/gemma-4-12B-it`, `judge_confidence_threshold: 0.85`,
`judge_top_k: 20`, `ratio_thresh: 2.0` -- all at the published values, none
adjusted after seeing a result. Machine-readable:
`/home/dcor/galbarak2/lment-ember-grid/judge_verdicts.json`.

## The accepted features -- all 27 cells judged

**11 of 27 cells accepted, 13 features.**

| rank | sp | seed | feature | conf | ratio_abs | the judge's description |
|---|---|---|---|---|---|---|
| **100** | **0.02** | **44** | **11** | **0.99** | **6.46** | **historical figures, locations and terminology associated with Ancient Rome** |
| 100 | 0.005 | 43 | 27 | 0.99 | 2.85 | the history, figures and geography of Ancient Rome |
| 300 | 0.005 | 42 | 254 | 0.98 | 3.59 | the Roman Empire and its historical figures, geography and military |
| 100 | 0.005 | 42 | 53 | 0.95 | **8.36** | the Roman Senate and its historical figures |
| 100 | 0.01 | 42 | 92 | 0.95 | 5.19 | history, culture and identity of Italy and the Roman Empire |
| 500 | 0.005 | 43 | 263 | 0.95 | 3.44 | military actions involving Roman legions and forces |
| 100 | 0.01 | 43 | 55 | 0.95 | 2.55 | Roman political titles, figures and leadership roles |
| 100 | 0.01 | 44 | 29 | 0.95 | 2.45 | Roman names, Latin roots |
| 300 | 0.005 | 44 | 287 | 0.95 | 2.15 | the siege of Rome during the Punic Wars |
| 300 | 0.02 | 44 | 288 | 0.95 | 2.06 | the assassination of Julius Caesar |
| 500 | 0.005 | 44 | 425 | 0.95 | 2.04 | the Crisis of the Third Century |
| 500 | 0.005 | 44 | 113 | 0.95 | 2.01 | a noisy collection of Roman history and Latin etymology |
| 500 | 0.005 | 43 | 210 | 0.85 | 2.27 | ancient invaders, architectural destruction |

## Where in the grid the concept lives

| axis | accept rate |
|---|---|
| **rank 100** | **6/9** |
| rank 300 | 3/9 |
| rank 500 | 2/9 |
| **g_sparsity 0.005** | **6/9** |
| g_sparsity 0.01 | 3/9 |
| g_sparsity 0.02 | 2/9 |
| seed 42 | 3/9 |
| seed 43 | 3/9 |
| seed 44 | 5/9 |

Both hyperparameters are cleanly monotone: **low rank and low sparsity find the
concept**, consistent with a small model holding it coarsely and concentrated
rather than finely divided.

*Correction to the 24-cell draft of this file.* On the partial grid the seed
looked like the dominant axis, swinging 1/7 to 5/8, and this document warned
that seed noise was as large as the hyperparameter effects. **That was an
artifact of which cells were missing** -- the three unjudged cells were the two
seed-42 champions and one seed-44 cell, and both seed-42 cells accepted. On the
full grid the seed spread is 3/9, 3/9, 5/9, clearly weaker than either
hyperparameter. The rank and sparsity trends are real; the seed panic was not.

## Both screens under-predict, and are wrong at the feature level too

Mean rank of a cell among all 27 (1 = the screen's best pick):

| screen | accepted cells | rejected cells |
|---|---|---|
| `screen_feature_grid.py` (marker list) | 11.8 | 15.5 |
| `token_distinctiveness.py` (data-driven) | 9.5 | 17.1 |

Token distinctiveness carries real signal at the cell level and the marker
screen barely does. But neither is usable as a chooser, and the calibration trio
showed the sharper failure: distinctiveness nominated cell `100:0.005:42` **for
feature 36**, its clean "king, monarchy, emperor, dynasty" candidate. The judge
**rejected feature 36** and accepted **feature 53** -- the Roman Senate --
instead. The screen picked the right cell for the wrong reason. Judging all 27
cells rather than a screen's top three is the only reason both the winner and
the highest-ratio feature in the grid were found at all.

## THE REPRODUCIBILITY PROBLEM

**The same cell, fitted twice, gives different features and opposite verdicts.**

Job 858233 -- the run whose failure motivated the grid -- used rank 100,
g_sparsity 0.01, seed 42. That is *the same cell* group g2 has now accepted
(feature 92, confidence 0.95). Every fit parameter matches: `max_iterations
20000`, `k_proj 30`, `g_sparsity 0.01`, fp32, seed 42. `set_seed()` **is** called
immediately before the embedding fit (`train_mf_features.py:206`), so this is not
a missing-seed bug.

The factorizations are genuinely different, not just the judge's wording:

    stats_embed.csv, feature 0     858233: num_concept=33  ratio_abs=2.787
                                   grid:   num_concept=21  ratio_abs=1.301

The two runs judged largely disjoint eligible sets -- 8 of ~23 features overlap
-- and **feature 92 was not eligible at all in the original run**, so no amount
of judge patience would have found it.

The visible difference is hardware: 858233 ran on an a6000 (n-601), the grid on
an RTX 3090 (n-301). The fit is an iterative solve with early stopping on
`no_improve`, so a small difference in matmul reduction order can move where it
halts and hence which sparsity pattern G converges to.

### Tested, and it is the hardware: deterministic per GPU model

Job **866925** re-fitted cell 100/0.01/42 on an RTX 3090 (n-307) -- a *different*
3090 node from the grid's (n-301), same everything else. The result is
**byte-identical**:

    md5 1441618c90ee611723778ae74f12f288   grid  n-301  stats_embed.csv
    md5 1441618c90ee611723778ae74f12f288   repro n-307  stats_embed.csv

So the fit is fully deterministic given the GPU *model*, and differs across
models. The seed is doing its job; the floating-point environment is not held
fixed by it.

The a6000 arm (866873) was **cancelled**, not run to completion: it sat 82
minutes in `D` state for 37 seconds of CPU while `/home/dcor` was serving about
1.4 MB/s, and it was starving the calibration judge sharing its node. It would
only have confirmed the complement -- that an a6000 re-fit reproduces 858233 --
and the 3090 arm's bit-identical result already establishes the mechanism. So
"deterministic per GPU model" rests on the 3090 pair; the a6000 side is inferred
from 858233 vs the grid differing, not separately replicated.

Replicate output: `/home/dcor/galbarak2/lment-ember-repro/r3090/`.

### What follows for the erasure run

`run_lment_ember` refits the factorization as part of its own flow. If it lands
on a different GPU model than the grid did, **it may not reproduce the feature
the judge accepted, and would then erase a feature nobody judged.** So the
erasure must run with `features.reuse` pointed at the grid's factorization under
`/home/dcor/galbarak2/lment-ember-grid/features/sp<sp>_seed<seed>`, not refit
from scratch. This is not an optimisation; it is what makes the run correspond to
the verdict above.

## The cell, chosen by the rule stated before the results were in

**rank 100 / g_sparsity 0.02 / seed 44, feature 11**
-- confidence 0.99, ratio_abs 6.46, "historical figures, locations and
terminology associated with Ancient Rome".

The rule, committed before the last three cells were judged: **judge confidence,
then `ratio_abs`**. Both are properties of the *control* model's own embedding
matrix. Neither can see the ablated twin, an accuracy, an erasure or a delta, so
this choice cannot be selection-on-the-outcome.

### The temptation to change the rule, and why it is refused

The final three cells produced **feature 53** (`100:0.005:42`, the Roman Senate)
at confidence 0.95 but `ratio_abs` **8.36** -- the highest in the whole grid, and
in the cell at the *better* sparsity (0.005 accepts 6/9; 0.02 accepts 2/9). A
`ratio_abs`-first rule would pick it, and there is a genuine argument for that
rule: the judge only ever emits confidences of 0.85, 0.95, 0.98, 0.99 and 1.0, so
it is a coarsely quantized quantity, while `ratio_abs` is a continuous
measurement. Keying on the quantized value and using the precise one only as a
tiebreak is arguably backwards.

**The rule is not changed, precisely because that argument only occurred to
anyone after seeing that it changes the answer.** Neither quantity touches the
ablated twin, so swapping would not be p-hacking in the strict sense -- but
rewriting a selection rule once the data reveal which candidate it favours is
the same species of error, and this project has already retracted two `acc_raw`
claims to it. The alternative is recorded here so the choice is auditable; it is
not taken.

Then `run_lment_ember` runs **once**. No further cells are judged afterwards to
look for a better erasure result.

### Reuse, not refit

Because the fit does not reproduce across GPU models, the run is given the
grid's own factorization via `lment.features.reuse: true` and a `cache_root`
pointing at the chosen cell, stamped with provenance by
`grid/publish_cell_features.py`. A refit could land on different hardware and
erase a feature nobody judged.

Constraining the erasure job to a 3090 would reproduce the fit bit-identically
and avoid all of this, but a 3090 has 24 GB and the run needs Gemma's 23 GB
beside the 1B. Reuse is the only route.

## Why the judge job kept failing, and what it was NOT

Three judge jobs failed with the same error before the calibration trio landed:

    JudgeProcessError: The judge worker did not answer within 3600s
                       (still running)

866583 on `gpu-h100-killable`/n-102, 866924 on `killable`/n-602. **Correcting an
earlier diagnosis in this repo's history: this is not GPU contention, and not an
h100 problem.** Attaching to the running job showed the judge worker in `D`
state -- uninterruptible I/O sleep -- with 10 seconds of CPU consumed in 52
minutes, absent from `nvidia-smi` entirely. It had never reached the GPU. It was
still reading the model.

The cause is the file: `google/gemma-4-12B-it` ships as **one unsharded 23.9 GB
`model.safetensors`**, so the load is a single sequential read with no shard
parallelism to exploit.

Measured 2026-09-08 from an idle client:

| source | throughput | implied Gemma load |
|---|---|---|
| `/home/dcor` | 28.0 MB/s | ~14 min |
| `/home/morg` | 28.8 MB/s | ~14 min |
| a contended compute node | ~7 MB/s | **~57 min** |

So **copying the model to `/home/morg` would buy nothing** -- the two filers are
within 3% of each other. (The recorded "`/home/morg` is 4.2x faster" compares it
to `/vol/scratch`, not to `/home/dcor`.) The variable is the client node, not the
filer, and 3600s was simply too close to the honest worst case.

The fix is therefore `startup_timeout_seconds: 10800` in the config, not a data
move and not a partition change. The read was never the problem; killing it at
one hour was.

**A self-inflicted part, recorded so it is not repeated.** 866924 failed partly
because the determinism replicate 866873 was scheduled onto the same node and
the two jobs read two large models over the same mount concurrently. Do not
co-schedule a judge job with anything else of ours that streams a model. The
four jobs that succeeded (866606-09, n-601, 1:08 each including the load *and*
5-7 cells) ran concurrently with each other but shared one already-warm page
cache, which is the opposite situation.

## Artifacts on disk

Not in git -- these are data, and the repo's convention is to reference results
by path rather than commit them (`ember_eval/results/` is likewise uncommitted).

| path (under `/home/dcor/galbarak2/`) | what |
|---|---|
| `lment-ember-grid/features/sp<sp>_seed<seed>/` | the 27 factorizations. The chosen cell, `sp0.02_seed44`, is stamped with provenance and is what the erasure reused. |
| `lment-ember-grid/judge_verdicts.json` | all 27 cells' verdicts, merged |
| `lment-ember-grid/judge_cells_{g1..g4,calib}.json` | the raw per-job outputs the merge was built from |
| `lment-ember-grid/screen.json` | `screen_feature_grid.py` output (marker screen) |
| `lment-ember-grid/distinctiveness.json` | `token_distinctiveness.py` output |
| `lment-ember-repro/r3090/` | the determinism replicate above |
| `lment-ember-grid/chosen_cell_verdict_backup/` | the chosen cell's `potential_features.csv` and `judge_trace.json`, copied before the erasure ran, since `select_with_judge` overwrites its own selection |

Jobs: features 858630; judging 866606-09 (24 cells) and 867066 (calibration trio,
after 866583 and 866924 both failed); determinism 866925; erasure 867391.

