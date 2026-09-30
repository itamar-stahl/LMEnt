# Validation report

Generated 2026-09-22 from `/home/morg/NLP_2526b/galbarak2/runs/nll_kl`, 
scored 2026-09-20/21. Every check below was executed, not asserted.


## Spec checklist

| # | Requirement | Result |
|---|---|---|
| 1 | Exact checkpoint recorded for every model | **PASS** — all 15 (topic, role) entries in `checkpoints.csv`, with resolved path, dtype, scoring timestamp and code revision |
| 2 | Only held-out test questions evaluated | **PASS** — every record carries `phase == "test"`; selection-phase items are excluded at load |
| 3 | Each expected cell contains 50 questions | **PASS** — 9 cells (3 topics x 3 groups), all exactly 50; 450 per-question rows total |
| 4 | Identical prompts and correct answers across models | **PASS** — for all 15 (topic, model) pairs the `completion_token_ids` and the stem's first token are identical to the full model's, item by item |
| 5 | NLL normalised by answer-token count | **PASS** — `nll == mean(token_nlls)` to 1e-9 for every record; answer lengths run 1-16 tokens (mean 3.36) |
| 6 | KL over the full vocabulary | **PASS** — saved distributions are 100,352 wide, matching the model vocabulary; a sampled row sums to 1.000000 |
| 7 | KL direction always twin -> comparison model | **PASS** — every KL is `KL(twin || model)`, twin's log-probs as the first argument, in a single code path |
| 8 | No NaN or infinite values | **PASS** — 0 non-finite values across 450 rows x all NLL/delta/KL columns |

## Additional consistency checks (not required, run anyway)

- **Table A + Table C = Table B.** The identity (M-T) + (T-F) = (M-F) holds across all 27 method-group cells; worst deviation **8.88e-16**.
- **SDs recomputed from `per_question.csv`** rather than reused from the aggregation pass; worst deviation **0.00e+00**.
- **Cross-checked against an independently generated table set** (`results/<topic>/results.json`, built by `report.py`): 45 overlapping KL cells agree to 5e-07.

## Checkpoints used

| Topic | Role | Model | Scored |
|---|---|---|---|
| Ancient Rome | Full | `lment-1b-control-2e-b131k` | 2026-09-20T18:34:52 |
| Ancient Rome | Twin | `lment-1b-norome-2e-b131k` | 2026-09-20T20:29:26 |
| Ancient Rome | EMBER | `ember_rome_d5000` | 2026-09-21T03:00:55 |
| Ancient Rome | RMU | `rmu_rome_L6hi_a10` | 2026-09-21T03:08:12 |
| Ancient Rome | SNMF | `snmf_rome_ratio_in` | 2026-09-21T03:15:24 |
| Baseball | Full | `lment-1b-control-2e-b131k` | 2026-09-20T18:35:04 |
| Baseball | Twin | `lment-1b-nobaseball-2e-b131k` | 2026-09-20T19:00:27 |
| Baseball | EMBER | `ember_baseball_d500` | 2026-09-21T02:39:10 |
| Baseball | RMU | `rmu_baseball_L5mid_a100` | 2026-09-21T02:46:23 |
| Baseball | SNMF | `snmf_baseball_ratio_out` | 2026-09-21T02:53:40 |
| Artificial Intelligence | Full | `lment-1b-control-2e-b131k` | 2026-09-20T18:35:15 |
| Artificial Intelligence | Twin | `lment-1b-noai-2e-b131k` | 2026-09-20T20:29:39 |
| Artificial Intelligence | EMBER | `ember_ai_d500` | 2026-09-21T02:24:19 |
| Artificial Intelligence | RMU | `rmu_ai_L5mid_a100` | 2026-09-21T02:24:38 |
| Artificial Intelligence | SNMF | `snmf_ai_ratio_in` | 2026-09-21T02:31:56 |

## Nothing is missing

All nine tables are complete. No model, checkpoint, question manifest or result was unavailable, and nothing was substituted.


## Two facts about provenance, recorded without interpretation

1. **The EMBER rows use the grid-selected checkpoint**, per the instruction to use only the selected final checkpoint. A second EMBER model exists for each topic — the released configuration (`lment-1b-rome-erased-b131k` and siblings) — which is *not* a product of this selection procedure and is therefore excluded from these tables. Its numbers are available in `../export_20260922/winners_and_references.csv` if wanted.

2. **KL direction was corrected on 2026-09-22.** Before that the twin-versus-full row was computed as `KL(full || twin)`. All KL values here are `KL(twin || model)`. Any KL figure predating 2026-09-22 is the reversed direction and should not be mixed with these.


## Scope limit carried from the run

One seed per condition; no result here is seed-replicated.

