# nll_kl export — 2026-09-22

Answer-token NLL and full-vocabulary KL for the concept-erasure study
(EMBER / RMU / SNMF vs the concept-excluded twins), exported for offline
analysis. Source run: `/home/morg/NLP_2526b/galbarak2/runs/nll_kl`, scored
2026-09-20/21.

**Start with `winners_and_references.csv`.** It is the table the paper needs:
every model that has both metrics, on the held-out test sets.

## Coverage — read this before asking why a cell is empty

| group | models | phases scored | NLL | KL |
|---|---|---|---|---|
| references (control, 3 twins, 3 released EMBER) | 7, each on all 3 topics | selection + test | yes | **yes** |
| selected winners (3 methods x 3 topics) | 9, each on its own topic | selection + test | yes | **yes** |
| non-selected grid candidates | 55 | **selection only** | yes | **no** |

The Sunday sweep covered all 64 candidates, but ran them with
`--no-dists --only-phase selection`: the selection rule uses answer NLL on the
selection half only, so candidates never needed next-token distributions to be
ranked. Distributions are 208 MB per model-topic and the scoring job is
essentially pure checkpoint I/O, so they were written only for the 7 references
and, afterwards, the 9 winners.

So for the 55 non-winners there is **no KL and no test-set number of any kind** —
only selection-half NLL. Getting KL across the whole grid means re-scoring those
55 with distributions: about 12 h of sequential GPU time (measured: 85 loads at
a median of 8 min, mean 13 min, max 63 min) and ~11 GB. It has to be sequential;
parallel readers starve each other on one node.

## Files

| file | rows | what |
|---|---|---|
| `winners_and_references.csv` | 90 | **the headline.** 30 (model, topic) pairs x 3 test sets, NLL and KL with SD and 95% CI |
| `per_item.csv` | 17,250 | every per-question number, all 85 (model, topic) pairs |
| `summary.csv` | 1,125 | per (model, topic, set, metric): n, mean, sd, ci95, median, frac_pos |
| `selection_grid.csv` | 64 | every candidate ranked by the selection rule, with its components |
| `questions.csv` | 900 | the item manifest: stem, completion, provenance, phase |

## Conventions

- **`nll_vs_control` / `nll_vs_twin`** are signed, `this model − reference`, paired
  per question. Positive means this model assigns the correct answer *less*
  probability than the reference does.
- **`kl_from_twin`** is `KL(twin ‖ model)`, the twin first, averaged over the
  answer positions. The twin is the common reference for every model so they are
  mutually comparable. KL is asymmetric and the direction matters: on Rome's
  target set the reversed form reads 0.761 where this one reads 0.955.
  (The published tables carried the reversed form on the twin-vs-full row until
  2026-09-22; that is corrected here and in `../results/`.)
- **`sd`** is across the 50 questions in the set — for NLL, the SD of the paired
  per-question differences.
- **`ci95`** is a bootstrap over questions, 1000 resamples, seed 0.
- n = 50 per set. Chance-level accuracy is not meaningful here; nothing is scored
  as a choice among options, only the supplied correct continuation.

## Caveats worth carrying into the analysis

- **The selection rule has no interior optimum.** `S = 0.5*delta_T -
  0.25*D_neighbour - 0.25*D_unrelated` is unbounded in edit size, so on Rome it
  selected `ember_rome_d5000`, which does ~17x the twin's neighbour damage. The
  `selected winner` EMBER rows characterise the rule, not the method. For EMBER
  itself use the `reference: <topic> EMBER released` rows.
- **One seed per cell.** No result here is seed-replicated.
- **Cross-concept rows are included and are useful**: each reference was scored on
  all three topics, so e.g. the Baseball-erased model on Rome questions is in
  here as an unrelated-model control (it sits 0.000001 nats from the control).
- Three checkpoints in the source run were silently written with unallocated
  blocks and two were scored before detection; they were rebuilt and the
  originals quarantined under `../_corrupt_20260921/`. Nothing corrupt feeds this
  export.

## Reproducing

Numbers here were cross-checked against the independently-generated tables in
`../results/<topic>/results.json`: 45 KL cells agree to 5e-07.

## Quick look — target set, held-out test questions

Answer NLL relative to each reference, and KL from the twin. `± sd` across the 50 questions. Full table in `winners_and_references.csv`; this is the target set only.


### rome

| model | NLL vs full | NLL vs twin | KL(twin ‖ model) |
|---|---|---|---|
| Full control | +0.000 ± 0.000 | -1.016 ± 1.001 | 0.955 ± 0.725 |
| Twin (the target) | +1.016 ± 1.001 | +0.000 ± 0.000 | 0.000 ± 0.000 |
| EMBER released | +2.402 ± 1.951 | +1.387 ± 1.953 | 2.162 ± 1.488 |
| EMBER (grid-selected) | +6.230 ± 3.775 | +5.214 ± 3.621 | 6.128 ± 2.868 |
| RMU (selected) | +0.713 ± 0.983 | -0.303 ± 0.977 | 1.143 ± 0.616 |
| SNMF (selected) | +0.138 ± 0.200 | -0.877 ± 0.953 | 0.835 ± 0.629 |

### baseball

| model | NLL vs full | NLL vs twin | KL(twin ‖ model) |
|---|---|---|---|
| Full control | +0.000 ± 0.000 | -0.377 ± 1.078 | 0.707 ± 0.495 |
| Twin (the target) | +0.377 ± 1.078 | +0.000 ± 0.000 | 0.000 ± 0.000 |
| EMBER released | +1.294 ± 1.922 | +0.918 ± 2.189 | 1.472 ± 1.258 |
| EMBER (grid-selected) | +1.514 ± 2.134 | +1.137 ± 2.338 | 1.715 ± 1.517 |
| RMU (selected) | +0.054 ± 0.133 | -0.322 ± 1.019 | 0.679 ± 0.467 |
| SNMF (selected) | +0.074 ± 0.297 | -0.303 ± 1.054 | 0.691 ± 0.477 |

### ai

| model | NLL vs full | NLL vs twin | KL(twin ‖ model) |
|---|---|---|---|
| Full control | +0.000 ± 0.000 | -0.595 ± 1.180 | 0.593 ± 0.380 |
| Twin (the target) | +0.595 ± 1.180 | +0.000 ± 0.000 | 0.000 ± 0.000 |
| EMBER released | +1.068 ± 1.562 | +0.473 ± 1.355 | 0.987 ± 0.873 |
| EMBER (grid-selected) | +2.220 ± 2.662 | +1.625 ± 2.507 | 1.963 ± 2.093 |
| RMU (selected) | +0.023 ± 0.108 | -0.573 ± 1.159 | 0.579 ± 0.364 |
| SNMF (selected) | +0.081 ± 0.182 | -0.514 ± 1.132 | 0.588 ± 0.370 |
