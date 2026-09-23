# Validation — accuracy-based hyperparameter selection (original unrelated set)

Every check below was executed against the exported files, not asserted.


## Required checks

| # | Requirement | Result |
|---|---|---|
| 1 | Exactly 64 candidate checkpoints considered | **PASS** — 64 in candidate_manifest.csv, all `load_status=scored`, 0 unscored |
| 2 | Exactly nine winners | **PASS** — 9 |
| 3 | 50 target + 50 neighbouring + 50 unrelated per candidate | **PASS** — 10050 per-question rows; 0 groups with a count other than 50 |
| 4 | Every question had exactly four scored options | **PASS** — 40200 per-option rows; 0 questions with a count other than 4 |
| 5 | Full model evaluated on identical questions | **PASS** — scored from the same frozen manifest, same question ids |
| 6 | Same model/question reproduces on rerun | **PASS** — the full model was scored twice in separate jobs on different nodes (main run; probe job 921076 on n-503) and returned **identical** unrelated counts: 14/50 Rome, 17/50 Baseball, 23/50 AI. Batch-invariance was separately tested at 9.5e-07 |
| 7 | No test question used in ranking | **PASS** — phases present in the ranked data: ['selection'] |
| 8 | No concept-excluded twin loaded or referenced | **PASS** — 0 rows reference any `no{rome,baseball,ai}` checkpoint; select_accuracy.py never opens one |
| 9 | No candidate selected via an old `selected=yes` field | **PASS** — rankings are recomputed from per-question data each run; the column is an output |
| 10 | No candidate omitted for failing to load | **PASS** — 0 omitted; all 64 integrity-checked before scoring |
| 11 | Score components reconstructible from the exports | **PASS** — recomputed `logprob_per_char` from `sum_logprob`/`continuation_char_count` for all 40200 options (0 mismatches); recomputed every prediction as argmax (0 mismatches) and every gold margin (0 mismatches) |
| 12 | Exported winner is rank 1 of its ranking | **PASS** — 0 winners not at rank 1 |
| 13 | No NaN or infinite values | **PASS** — 0 non-finite scores |

## Full-model baseline and the normalisation denominator

`Acc~ = clip01((Acc - 0.25)/(Acc_F - 0.25))`, so `Acc_F - 0.25` must be safely positive.

| topic | target | neighbour | unrelated | smallest denominator |
|---|---|---|---|---|
| rome | 0.560 | 0.480 | 0.280 | **+0.030** |
| baseball | 0.580 | 0.440 | 0.340 | **+0.090** |
| ai | 0.440 | 0.540 | 0.460 | **+0.190** |

## Selected checkpoints

| topic | method | winner | H | preservation | efficacy |
|---|---|---|---|---|---|
| ai | EMBER | `ember_ai_d500` | 0.7230 | 0.8453 | 0.6316 |
| ai | RMU | `rmu_ai_L6hi_a10` | 0.5430 | 0.7645 | 0.4211 |
| ai | SNMF | `snmf_ai_ratio_in` | 0.4740 | 0.9500 | 0.3158 |
| baseball | EMBER | `ember_baseball_d10` | 0.7595 | 0.8824 | 0.6667 |
| baseball | RMU | `rmu_baseball_L6hi_a10` | 0.3803 | 0.8824 | 0.2424 |
| baseball | SNMF | `snmf_baseball_ratio_both` | 0.0000 | 0.8824 | 0.0000 |
| rome | EMBER | `ember_rome_d200` | 0.6815 | 0.7222 | 0.6452 |
| rome | RMU | `rmu_rome_L6hi_a10` | 0.2218 | 0.7895 | 0.1290 |
| rome | SNMF | `snmf_rome_ratio_out` | 0.0000 | 1.0000 | 0.0000 |

## Caveats

- **Two cells are not determined by the rule.** In `rome/SNMF` and `baseball/SNMF` every candidate scores H = 0.0000: normalised target accuracy clips to 1.0, so efficacy is exactly zero and the harmonic mean collapses. Those models did not reduce target accuracy at all. The listed winner there comes from the tie-break, not from any measured difference, and should be read as *undetermined* rather than as a selection.
- **The tie-break order for RMU and SNMF is declared, not inherited.** The repository defines an aggressiveness order for EMBER (smaller delta) but not for the other two; ours is recorded in `rankings.csv` as `tiebreak_rule` and flagged there as a declared convention.
- **One seed per condition.** Nothing here is seed-replicated.
- **Cross-check against the sciq unrelated set: all nine winners are identical.** The choice of unrelated set changes H slightly but no selection.

