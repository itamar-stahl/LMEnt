# The EMBER hyperparameter grid for Ancient Rome

Why this directory exists, what the first run found, and how to read the result.

## What happened first

Job **858233**, 2026-09-06, standalone EMBER on the Ancient Rome pair's control
(`hf-models/lment-1b-control-2e-b131k`, job 853707, step 54832). It failed:

    ValueError: The feature judge selected no features above the confidence
    threshold; trace saved to .../judge_trace.json

That is not a crash, and not a configuration mistake. It is a measurement.

**Rome's embedding signature is much stronger than Pornography's.** The earlier
Pornography attempt (job 791515, 2026-08-28, on the retired pair's control) died
one step sooner, at the ratio prefilter, because no feature reached
`ratio_thresh: 2.0`:

| | Pornography | **Ancient Rome** |
|---|---|---|
| max `ratio_abs` | 1.6889 | **7.9673** |
| median | 0.9801 | 1.4065 |
| features clearing 2.0 | **0** | **24** |

Rome's corpus footprint is 26x Pornography's (65,844 chunks, 0.628%, against
2,546), and it shows exactly where you would predict.

**But the judge rejected all 24, at confidence 0.95-1.0, and it was right.**
Gemma-4-12B read each feature's activating tokens and described them:

| feature | `ratio_abs` | what it actually is |
|---|---|---|
| 22 | 7.97 | negation, limitation, scarcity, restriction |
| 45 | 6.07 | food, culinary items, dining |
| 10 | 5.01 | armed conflict, violent upheaval |
| 0 | 2.79 | collective entities and organised social structures |
| 4, 43, 49 | | suffixes, past-tense verbs, numerical prefixes |
| 61 | | numerical roots and Latin-based linguistics |

Every one is a **generic category that Roman prose activates** — governance,
conflict, aqueducts, Latin morphology — rather than a feature meaning "Ancient
Rome". The judge's prompt tells it to reject anything "broad like sport for
basketball", and it applied that correctly.

Note how many are pure morphology. That is characteristic of a 1B
Wikipedia-only model, whose embedding matrix spends much of its capacity on
surface form. EMBER was developed on `gemma-2-2b-it` and
`Llama-3.1-8B-Instruct`, which carry far more semantic structure in embedding
space.

**So the open question is not "did the run work" but: does ANY (rank,
g_sparsity) isolate a Rome direction in this model?** One failed setting cannot
answer that. A grid can, and either answer is worth reporting.

## The two stages

Measured from 858233's `timing.json`, the two halves of a cell differ by three
orders of magnitude:

| | cost |
|---|---|
| factorization | **6.55 s** |
| Gemma judge | most of that run's **49m30s** (23 GB load, 2 generations per eligible feature) |

A 27-cell grid with the judge inside the loop is ~20 hours. Without it, 3
minutes. So:

    stage 1   run_feature_grid.slurm      build features for every cell
    stage 2   screen_feature_grid.py      rank cells, judge-free
    stage 3   the judge, on the survivors only

## The axes

| axis | values | why |
|---|---|---|
| `rank` | 100, 300, 500 | 100 components over \|V'\|~5,100 may be too coarse, pooling several senses per feature. d_model is 2,048, so 500 is still sane. |
| `g_sparsity` | 0.005, 0.01, 0.02 | tokens kept per feature |
| `seed` | 42, 43, 44 | **not optional** — see below |

**`g_sparsity` is a fraction of `|V'|`, not of the model vocabulary.** `|V'|` is
the vocabulary of the concept+neutral sentences, about 5,100 tokens — not the
model's 100,352. Measured from 858233's `token_features.csv`, `0.01` gives **51
tokens per feature**, so:

| `g_sparsity` | tokens/feature |
|---|---|
| 0.005 | ~25 |
| 0.01 | 51 (the failed run) |
| 0.02 | ~102 |

Below ~0.005 a feature is too thin to interpret at all.

**Seed is an axis because the factorization may be hitting poor local optima.**
858233's loss sat at `15848.176758` from iteration ~150 to its early stop at
578 — a fast plateau from `init="random"`. With one seed per cell there is no
way to separate "this cell is bad" from "this cell's init was bad".
`SparseMatrixFactorization` implements `init_svd` and `init_knn`, but no config
key reaches them; that is a real gap and a candidate fix if the seed spread
turns out to be large.

## Why the screen does not use `ratio_abs`

`stats_embed.csv` carries `ratio_abs`, and `token_features.csv` carries
`num_concept_related`. Both look like ready-made specificity scores. **Neither
is**, and 858233 proves it:

    feature 22:  ratio_abs 7.97,  45 of 51 tokens "concept related" (88%)
                 judge: "negation, limitation, scarcity, restriction"

Over all 100 features, `corr(ratio_abs, concept fraction) = 0.84`. They are one
signal, and that signal is **"appears in Roman prose"**.

A token is marked "concept related" if it occurred in any concept sentence — and
Roman Wikipedia articles are written in ordinary English. So `not`, `without`,
`food` and `organizations` all qualify, and ~88% of any feature's tokens do.
The statistic cannot tell a feature about Rome from a feature about negation
that happens to appear in Rome articles. That is precisely why 24 features
cleared the filter and all 24 failed the judge.

`screen_feature_grid.py` uses **Rome-distinctive vocabulary** instead: the
`ROME_MARKERS` alternation copied verbatim from
`ember_eval/run_rome_heldout.slurm`, where it proved `chunk_id -> Roman text`
before the held-out measurement was trusted. `caesar`, `legion*`, `aqueduct*`,
`denarius`, `carthag*` — words that are *about* Rome, not merely near it.

**Validated against the one cell whose answer is known.** Run on 858233's own
output, the screen reports 0 features both Roman-looking and past the prefilter,
best marker share 5.9% — agreeing with the judge's verdict on all 24.

The screen is a proxy, not a replacement. It reads word lists, not meanings. A
high score means "worth 45 minutes of judge"; only the judge decides whether a
feature is about Rome.

## The rule for choosing a cell

**Choose on feature quality, which never sees the ablated twin.** Every number
in stage 1 and 2 comes from the control model's own embedding matrix — no
accuracy, no erasure, no delta, no ablated twin. So the choice cannot be
influenced by the size of the effect the erasure is meant to show.

**Never choose the cell that makes the erasure damage Rome most.** That number
*is* the result; picking 1 of 27 settings by it manufactures an effect whether
or not one exists. This project has already paid for that mistake once — two
claims made on `acc_raw` were retracted in `ember_eval/EVALUATION.md`, and
`completion_eval/metric_bakeoff.py` was written afterwards precisely to rank
scoring rules on criteria "independent of the ablation".

Fix the cell in stage 2, then run the erasure once. That result is then a real
test.

## If no cell works

Then the finding is that **a sparse factorization of this model's embedding
matrix does not isolate an Ancient Rome direction at any of these settings** —
a genuine result about 1B embedding geometry, and far stronger than the single
failed setting 858233 gave. It also bears directly on the erasure work: EMBER
assumes a concept has a findable embedding subspace, and that assumption may
simply not hold at this scale.

Before concluding it, note the untested axis: **the concept sentences are
EMBER's stock Rome Wikipedia text, not the 56 QIDs / 65,844 chunks actually held
out of training.** So the features are defined against a different Rome than the
one that was ablated. `Ember-on-LMEnt/README.md` names this gap: "the LMEnt
chunk-based alternative remains a future sampling seam; it is not part of the
current flow." That may matter more than rank or sparsity.

## Running it

    sbatch grid/run_feature_grid.slurm          # stage 1, ~9 model loads
    python grid/screen_feature_grid.py \
        --grid-root /home/dcor/galbarak2/lment-ember-grid/features \
        --json-out /home/dcor/galbarak2/lment-ember-grid/screen.json

`RANKS`, `SPARSITIES` and `SEEDS` are environment overrides on the slurm script.
Outputs live on `/home/dcor`, not in the repo.
