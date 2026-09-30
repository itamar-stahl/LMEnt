# Does any erasure method reach the state of never having trained on the concept?

**No.** Across three concepts and three erasure methods, **the twin is bracketed
but never hit.** EMBER overshoots it on every concept, under both its released
configuration and the one this grid selected; RMU and SNMF fall short of it on
every concept. All twelve of those model-by-concept contrasts have a 95% CI
excluding zero.

`ERASURE_RESULTS.md` answered this for one concept, one erasure configuration and
one metric, and found EMBER overshooting by 1.7-2.2x. This generalises it to
three concepts, three methods, and a different instrument — answer-token NLL and
KL rather than `pmi_per_char` on completion options — and the overshoot
reproduces at 2.4x on Rome.

Run 2026-09-21. Code in `ember_eval/nll_kl/`, outputs under
`/home/morg/NLP_2526b/galbarak2/runs/nll_kl`.

## The instrument

For each concept we build six sets of 50 question-completion pairs — **target**
(the concept), **neighbour** (its adjacent domain), **unrelated** (drawn from 17
other topics) — each split into a **selection** half and a **test** half
(`build_sets.py`, seed 20260920). Checkpoints are chosen on the selection half
and reported on the test half, and the rule is fixed in `select_checkpoint.py`
before any test number is looked at.

Each model is scored on the answer tokens only: mean NLL per token of the gold
completion given the stem, plus the full next-token distribution at each answer
position, which gives KL against a reference. Two metrics, three sets, two
phases.

**KL's reference is always the twin** — `KL(twin ‖ model)` — so every model is
asked the same question: how closely does it reproduce the twin, with deviations
weighted by the twin's own distribution. KL is asymmetric and the choice is not
cosmetic: on Rome's target set the reversed form reads 0.761 where this one reads
0.955, a 26% difference, and reversing can reorder models as well as rescale
them. The NLL difference keeps its own reference, named per row, because it is
signed and the reference is what fixes the direction of the sign.

Scoring is **100% I/O**. The model load takes 7-63 minutes; the scoring itself
takes 1 second. Plan any sweep around the reads, not the compute.

## The ceiling: what never having learned actually costs

The ablated twin is the reference the erasures are trying to reach. It is a
clean one — the concept moves, the neighbourhood does not:

| Twin vs Full, test sets | Target | Neighbour | Unrelated |
|---|---|---|---|
| Ancient Rome | **+1.016** [+0.742, +1.297] | +0.171 [-0.003, +0.351] | +0.031 [-0.109, +0.166] |
| Baseball | **+0.377** [+0.070, +0.671] | +0.093 [-0.108, +0.294] | -0.015 [-0.160, +0.137] |
| Artificial intelligence | **+0.595** [+0.293, +0.930] | -0.139 [-0.363, +0.070] | +0.146 [-0.026, +0.345] |

Answer NLL difference in nats/token, evaluated − reference, n = 50 per cell.
Target CIs exclude zero on all three; neighbour and unrelated CIs straddle zero
on all three. That is the shape a successful erasure should reproduce.

Note the size ordering: Rome costs 2.7x what Baseball costs. Any cross-concept
comparison of erasure strength has to be read against that, not in raw nats.

## Nobody lands on the twin

Distance from the twin on the target set — positive is overshoot, negative is
shortfall:

| vs Twin, target, test | Rome | Baseball | AI |
|---|---|---|---|
| EMBER (selected) | **+5.214** [+4.235, +6.202] | **+1.137** [+0.520, +1.774] | **+1.625** [+0.977, +2.264] |
| EMBER (released) | **+1.387** [+0.903, +1.911] | **+0.918** [+0.323, +1.518] | **+0.473** [+0.103, +0.865] |
| RMU | **-0.303** [-0.557, -0.038] | **-0.322** [-0.599, -0.033] | **-0.573** [-0.906, -0.265] |
| SNMF | **-0.877** [-1.146, -0.621] | **-0.303** [-0.585, -0.007] | **-0.514** [-0.827, -0.217] |

Twelve contrasts, twelve CIs excluding zero, and the sign is determined entirely
by the method: EMBER above, RMU and SNMF below, on every concept. This is not a
tuning problem that a finer grid would fix — the two families miss in opposite
directions.

KL says the same thing without needing a sign. Distance from the twin on the
target set, with the full model's own distance as the baseline any erasure has to
beat by moving *below* it:

| KL(twin ‖ model), target, test | Rome | Baseball | AI |
|---|---|---|---|
| Full — no erasure at all | 0.955 | 0.707 | 0.593 |
| EMBER (selected) | **6.128** | **1.715** | **1.964** |
| EMBER (released) | **2.162** | **1.472** | **0.987** |
| RMU | 1.143 | 0.679 | 0.579 |
| SNMF | 0.835 | 0.691 | 0.588 |

Every EMBER cell is *further* from the twin than not erasing at all — on Rome the
selected one is 6.4x as far. RMU and SNMF sit within noise of the full model on
all three concepts; the largest move is SNMF on Rome at 0.12 nats closer, with
the two CIs overlapping over most of their range. Neither metric finds an erasure
that lands on the twin, and they do not disagree about which way anything moved.

### EMBER overshoots, and the selection rule cannot stop it

Against the full control, the released EMBER models remove 2.4x (Rome), 3.4x
(Baseball) and 1.8x (AI) what the ablation removed. The Rome figure is the one to
check against `ERASURE_RESULTS.md`, which measured 1.74x on train items and 2.23x
on test items with a completely different scoring rule. **2.4x here against 2.2x
there is the cross-instrument agreement this document rests on.** If that ever
moves, one of the two instruments changed.

The *selected* checkpoints are worse, because the rule

    S = 0.5*delta_T - 0.25*D_neighbour - 0.25*D_unrelated

has no interior optimum. It is unbounded in delta: a larger edit always wins
while the target degrades faster than half the collateral, and it does,
indefinitely. Rome's ranking stayed strictly monotone after the grid was extended
ten-fold to delta = 5000, which is what the rule then picked:

| Rome EMBER vs Full, test | Target | Neighbour | Unrelated |
|---|---|---|---|
| twin (the goal) | +1.016 | +0.171 | +0.031 |
| released | +2.402 | +0.934 | +0.106 |
| **selected, d5000** | **+6.230** | **+2.905** | **+0.597** |

d5000 does **17x** the neighbour damage of never having learned Rome. It is not
an erasure, it is a broken model that the rule ranks first. Baseball fails the
same test from the other side: d5 through d500 are flat within noise (S = 0.866
to 0.982), so the rule picks the grid edge there too.

**Fix the rule before spending more GPU on this grid.** The 0.25 weights cannot
punish collateral fast enough to produce an interior maximum, so re-running as-is
reproduces the same ranking. Two candidate repairs, neither tried: constrain
D_neighbour to the twin's own value rather than penalising it linearly, or select
on distance-to-twin directly instead of a weighted sum.

### RMU and SNMF do not move the target at all

Against the full control, on the target set:

| vs Full, target, test | Rome | Baseball | AI |
|---|---|---|---|
| RMU | +0.713 [+0.427, +0.979] | +0.054 [+0.018, +0.093] | +0.023 [-0.007, +0.051] |
| SNMF | +0.138 [+0.087, +0.192] | +0.074 [-0.007, +0.154] | +0.081 [+0.034, +0.128] |

On Baseball and AI these are hundredths of a nat against twin costs of 0.377 and
0.595. That is nothing, and it replicates the null in
`mlp_erasure/DELTA_FIX_RERUN_RESULTS.md` — reached there on multiple-choice
accuracy, here on answer NLL, so the two agree across instruments as well as
across concepts.

Rome RMU is the only cell that moves at all, and it moves the wrong thing:
**+1.019 on the neighbour set against +0.713 on the target**. It damages the
neighbourhood more than the concept. KL says the same thing about the collateral,
though it cannot speak to direction: measured from the twin, RMU sits at 1.143 on
target and 0.628 next door, where the full control sits at 0.955 and 0.253. On
the concept RMU is 1.2x as far from the twin as doing nothing at all; next door,
2.5x.

The selection rule's own scores say this plainly — four of the six RMU/SNMF cells
have a **negative** winning S, meaning the best candidate in the grid is worse
than doing nothing:

| best S | Rome | Baseball | AI |
|---|---|---|---|
| RMU | +0.087 | **-0.027** | **-0.007** |
| SNMF | +0.021 | **-0.043** | **-0.005** |

And two of the three RMU winners — `rmu_baseball_L5mid_a100` and
`rmu_ai_L5mid_a100` — had **failed their own build-time
`rotation_is_substantial` sanity gate**. The rule is ranking noise and landing on
cells the builder itself flagged as never having rotated the representation.

## What is still not established

- **Nothing here says why the two families miss in opposite directions.** EMBER
  edits embedding rows, RMU and SNMF edit MLP cells; that is a plausible story
  and this document does not test it.
- **The neighbour sets are not validated as neighbours.** They are the adjacent
  domain by construction, not by any measured similarity to the target.
- **One seed per cell, so the SNMF rows do not yet count.**
  `mlp_erasure/DELTA_FIX_RERUN_RESULTS.md` sets a standing rule: no SNMF result
  counts until it reproduces across independent factorization seeds, three
  Baseball effects having already died on that check. Nothing here is
  seed-replicated. The rule cuts the right way for a null — an effect that is
  absent under one seed is weak evidence, not strong — but a *positive* cell in
  this table would need seeds before it could be cited, and the same caution
  belongs on RMU, whose numbers are equally small.
- **The selected EMBER checkpoints are artefacts of a broken rule**, so the
  "EMBER (selected)" rows characterise the rule, not the method. The "EMBER
  (released)" rows are the ones to cite for EMBER itself.
- **KL is reported but not leaned on.** Both metrics agree everywhere they are
  compared here, so nothing turns on the choice. They are not independent
  evidence either: both read the same model on the same 50 completions, so
  agreement between them is not replication.

## The data this rests on

71 models scored: 7 references (control, three ablated twins, three released
erased models) at 300 records with distributions, plus 64 candidates — 30 EMBER
deltas, 24 RMU cells, 10 SNMF variants — at 150 records each, with the 9 selected
winners re-scored to 300 with distributions.

**Three checkpoints were silently corrupt and two of them were scored and
ranked** before detection. `model-00001-of-00002.safetensors` had the correct
apparent size but 380-710 MB of blocks were never allocated, and read back as
zeros; every build job reported COMPLETED. The tell is degradation of the
*unrelated* set — `rmu_rome_L6hi_a100` read 7.250 / 7.669 / 8.775 against a
3.241 / 2.702 / 5.059 control, and 3.560 / 2.988 / 5.091 after rebuild. Cause
was concurrency, not a bad node: n-306 and n-202 each wrote one corrupt and one
healthy model in the same window. Originals are kept under
`runs/nll_kl/_corrupt_20260921/` with a full account.

`check_alloc.py` now refuses a short checkpoint at scoring time, and
`repair_holed.slurm` serialises rebuilds and verifies each write. The rebuilds
reproduced every RMU sanity metric to six significant figures
(`edited_rel_change` 0.051395 → 0.051395), which is what establishes that only
the write was ever broken.

## Reproducing and re-checking

    python ember_eval/nll_kl/audit.py --run <run root>

asserts the six properties a finished run has — every candidate scored, every
reference complete, all 9 selections ranking every candidate, winners test-scored
with distributions, no holed checkpoints, and quarantined originals genuinely
superseded. It exits non-zero and names what is wrong otherwise. It exists
because on 2026-09-20 three different things were broken at once and every job
involved reported success.

    python ember_eval/nll_kl/finalize.py --run <run root> --root <worktree>

re-derives everything downstream of the raw scores and is idempotent: it checks
the disk before each step, so it can be killed anywhere and re-run. `--no-gpu`
restricts it to what needs no model load.
