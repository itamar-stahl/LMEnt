# Current per-question NLL and KL — accuracy-selected checkpoints

The full `report.py` output for the nine checkpoints chosen by the accuracy
rule in `ember_eval/acc_selection/results_sciq/selected_checkpoints.json`:

| concept | EMBER | RMU | SNMF |
|---|---|---|---|
| Ancient Rome | `ember_rome_d200` | `rmu_rome_L6hi_a10` | `snmf_rome_ratio_out` |
| Baseball | `ember_baseball_d10` | `rmu_baseball_L6hi_a10` | `snmf_baseball_ratio_both` |
| Artificial intelligence | `ember_ai_d500` | `rmu_ai_L6hi_a10` | `snmf_ai_ratio_in` |

plus the full model (`lment-1b-control-2e-b131k`), each concept's twin, and the
released EMBER checkpoint as a reference row.

`<topic>/table.md` here is the same file as `../tables_accwinners/<topic>/table.md`;
this directory holds the `results.json` that was missing beside it, which carries
every per-item value the tables summarise.

## Do not use instead

- `../paper_tables/per_question.csv` and `../export_20260922/` are the **answer-NLL**
  selection. Six of the nine cells chose a different checkpoint.
- `../results/<topic>/results.json` in the run root is the same superseded round.

## What is in each results.json

- `rows` — the signed NLL-difference comparisons (`Twin vs Full`, each method
  `vs Twin` and `vs Full`).
- `kl_rows` — one row per evaluated model (`Full`, `EMBER`, `RMU`, `SNMF`,
  `EMBER-released`), all of them `KL(twin ‖ model)` with the twin on the left.
- `items` under every row — per question: `id`, `set`, `role`, `phase`,
  `n_tokens`, `nll_eval`, `nll_ref`, `delta`, and `kl` where distributions
  exist.

Six groups of 50 per row: `{target,neighbour,unrelated} × {selection,test}`.
KL is present on the three **test** groups only — `score_model.py` saves the
full next-token distributions for test-phase items, which is all the protocol
needs and 217 MB per model as it is.

The question ids are the same ids the accuracy export uses, so
`../../acc_selection/final/current_test_per_question.csv` joins to these
row-for-row.

## Reproducing

`run_manifest.json` carries the exact commands, the input trees, output
checksums, the NLL/KL/accuracy definitions and the result of all twelve
validation checks. The checks are executable:

    python ember_eval/nll_kl/validate_accwinners.py --out /tmp/validation.json
