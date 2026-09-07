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

## The accepted features

| rank | sparsity | seed | feature | conf | ratio_abs | the judge's description |
|---|---|---|---|---|---|---|
| 100 | 0.02 | 44 | 11 | **0.99** | **6.46** | historical figures, locations and terminology associated with Ancient Rome |
| 100 | 0.005 | 43 | 27 | **0.99** | 2.85 | the history, figures and geography of Ancient Rome |
| 100 | 0.01 | 42 | 92 | 0.95 | 5.19 | history, culture and identity of Italy and the Roman Empire |
| 500 | 0.005 | 43 | 263 | 0.95 | 3.44 | historical military actions involving Roman legions and forces |
| 100 | 0.01 | 43 | 55 | 0.95 | 2.55 | Roman political titles, figures and leadership roles |
| 100 | 0.01 | 44 | 29 | 0.95 | 2.45 | Roman names, Latin roots, geographical/historical markers |
| 300 | 0.005 | 44 | 287 | 0.95 | 2.15 | the siege of Rome during the Punic Wars |
| 300 | 0.02 | 44 | 288 | 0.95 | 2.06 | the assassination of Julius Caesar |
| 500 | 0.005 | 44 | 425 | 0.95 | 2.04 | the Crisis of the Third Century and the Roman Empire |
| 500 | 0.005 | 44 | 113 | 0.95 | 2.01 | a noisy collection of fragments related to Roman history and Latin etymology |
| 500 | 0.005 | 43 | 210 | 0.85 | 2.27 | historical military conquest involving ancient invaders |

Three cells are not yet judged -- `100:0.005:42`, `300:0.005:42`, `500:0.02:44`,
the two screens' champions and their shared worst -- because job 866583 failed;
see "The h100 lesson" below.

## Where in the grid the concept lives

| axis | accept rate |
|---|---|
| rank 100 | **5/8** |
| rank 300 | 2/8 |
| rank 500 | 2/8 |
| g_sparsity 0.005 | **4/7** |
| g_sparsity 0.01 | 3/9 |
| g_sparsity 0.02 | 2/8 |
| seed 42 | **1/7** |
| seed 43 | 3/9 |
| seed 44 | **5/8** |

Low rank wins, and low sparsity helps -- both consistent with a small model
holding a coarse, concentrated concept rather than a finely divided one. **The
seed swing from 1/7 to 5/8 is the number to worry about**: it is as large as the
effect of the hyperparameters the grid was built to explore, which means no
single cell's verdict should be read as a property of its rank and sparsity.

## Both screens failed as predictors -- do not reuse them to rank

Mean rank of a cell among all 27 (1 = the screen's best pick):

| screen | accepted cells | rejected cells |
|---|---|---|
| `screen_feature_grid.py` (marker list) | 13.1 | 14.7 |
| `token_distinctiveness.py` (data-driven) | **10.0** | **16.5** |

The marker screen is indistinguishable from noise, exactly as its own docstring
feared. Token distinctiveness carries real signal but is not usable as a chooser:
its #2 cell was **rejected** and its #26 cell was **accepted**. Judging all 27
rather than the screens' top three was the right call, and that decision is the
only reason the accepted cells were found at all -- a top-three-by-screen plan
would have paid for `100:0.005:44` (rejected) and missed `100:0.02:44`, the
highest-confidence, highest-ratio feature in the grid.

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
halts and hence which sparsity pattern G converges to. Jobs 866873 (a6000) and
866925 (3090) re-fit that one cell to test this; see below.

### What follows for the erasure run

`run_lment_ember` refits the factorization as part of its own flow. If it lands
on a different GPU model than the grid did, **it may not reproduce the feature
the judge accepted, and would then erase a feature nobody judged.** So the
erasure must run with `features.reuse` pointed at the grid's factorization under
`/home/dcor/galbarak2/lment-ember-grid/features/sp<sp>_seed<seed>`, not refit
from scratch. This is not an optimisation; it is what makes the run correspond to
the verdict above.

## How the cell will be chosen, stated before choosing

One cell, chosen on **judge confidence, then `ratio_abs`** -- both properties of
the *control* model's own embedding matrix. Neither quantity can see the ablated
twin, an accuracy, an erasure or a delta, so this choice cannot be
selection-on-the-outcome. Then `run_lment_ember` runs **once**. No further cells
are judged afterwards to look for a better erasure result; that is the failure
that retracted two `acc_raw` claims (`ember_eval/EVALUATION.md`).

On the 24 cells judged so far that rule selects **rank 100 / g_sparsity 0.02 /
seed 44, feature 11** -- highest confidence (0.99) and highest ratio_abs (6.46)
in the grid, with the most squarely on-concept description. The three pending
cells could displace it only by producing a confidence above 0.99.

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
