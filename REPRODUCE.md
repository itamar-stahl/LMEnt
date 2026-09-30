# Reading and checking this repository

This repository holds the code, data and results behind
[`paper/`](paper/) — *Can Concept Erasure Reproduce Concept Exclusion? A Matched
Evaluation of EMBER, RMU, and SNMF*.

It is built on the LMEnt suite, whose own code and README it also contains. This
file is the shortest path from a fresh clone to understanding what was done and
checking that the paper's numbers are real.

## Start here

1. **[`paper/`](paper/)** — `main.tex`, `sections/`, `appendices/`. Build with
   `latexmk -pdf main.tex` from that directory.
2. **[`README.md`](README.md)** — setup, the repository layout map, and the
   LMEnt suite's own documentation.
3. **This file** — where each number in the paper comes from.
4. **[`archive/`](archive/)** — work that is *not* in the paper, moved aside and
   indexed in [`archive/README.md`](archive/README.md). Nothing in the paper
   depends on it.

## The study in one paragraph

One 1B OLMo-2 model was trained on the full LMEnt Wikipedia corpus, and three
*twins* were trained identically except that loss was masked on chunks linked to
one concept each (Ancient Rome, Baseball, artificial intelligence). Five
post-training erasure conditions (EMBER, RMU, SNMF, RMU+EMBER, SNMF+EMBER) were
then applied to copies of the full model. The question is whether erasure
*resembles the twin*, not merely whether it suppresses the concept.

## Pipeline, in order

| Directory | What it does |
|---|---|
| [`retrieval-index/`](retrieval-index/) | Builds the Elasticsearch entity index over the pretraining corpus |
| [`aggregate_entity_annotations/`](aggregate_entity_annotations/) | Merges ReFinED entity links and Maverick coreference into per-chunk annotations |
| [`Untaught/`](Untaught/) | Trains the full model and the concept-excluded twins. Also the environment layer — `Untaught/framework/env.sh` sets `PYTHONPATH` and `OLMO_CORE_SRC` |
| [`Ember-on-LMEnt/`](Ember-on-LMEnt/) | EMBER embedding-level erasure, the concept/question data, and the edit-strength grid |
| [`mlp_erasure/`](mlp_erasure/) | RMU and SNMF, the MLP-level methods |
| [`ember_eval/`](ember_eval/) | All evaluation: checkpoint selection, accuracy, NLL, KL, bootstrap intervals |
| [`third_party/`](third_party/) | Vendored forks of dolma, OLMo-core, olmes, ReFinED, maverick-coref |

## Where every number in the paper lives

All of these are committed. You do not need the models or the cluster to check
any of them.

| Paper location | File |
|---|---|
| Table 1, concept-exclusion sets (QIDs, chunks, share) | `Untaught/blacklists/{ancient_rome_core,baseball_core_teams,ai_core}.json` for the QID lists; chunk counts and shares in `Untaught/model_cards/lment-1b-no*-2e-b131k.README.md` |
| **Table 2**, `H_test` column | `ember_eval/ensembles/results/panel_A_accuracy_sciq.csv` — 21 rows, all seven models × three concepts, including the twins |
| **Table 2**, `R_abs` column | `ember_eval/ensembles/results/summary.csv` |
| **Table 2**, `R_KL` column | `ember_eval/nll_kl/rkl_bootstrap.csv` |
| Figure 1, both panels and their error bars | Same two files — they carry the 95% CI columns |
| App. Checkpoint Selection, candidate grid (97) | `ember_eval/acc_selection/results_sciq/candidate_manifest.csv` (64 standalone) and `ember_eval/ensembles/results/rankings.csv` (33 ensemble) |
| App. Checkpoint Selection, selected configurations | `ember_eval/acc_selection/results_sciq/selected_checkpoints.json` and `ember_eval/ensembles/results/selected_checkpoints.json` |
| App. Complete Results, Panel A (accuracies) | `ember_eval/ensembles/results/panel_A_accuracy_sciq.csv` |
| App. Complete Results, Panel B/C — target and neighbouring NLL/KL | `ember_eval/nll_kl/results_accwinners/{rome,baseball,ai}/{results.json,table.md}` |
| App. Complete Results, Panel B/C — **SciQ** rows | standalone methods: `ember_eval/acc_selection/final/sciq_nll_kl.csv`; ensembles: `ember_eval/ensembles/results/nll_kl_summary.csv` |
| App. Diagnostics, `P_closer` and its CI | `ember_eval/ensembles/results/summary.csv` |
| App. Diagnostics, `S_NLL` selectivity contrast | `ember_eval/ensembles/results/nll_selectivity.csv` |
| App. Evaluation Data, the frozen question sets | `ember_eval/nll_kl/sets/{rome,baseball,ai,sciq}.json` |

### One trap, because it will confuse you

The name `unrelated` refers to **two different question sets** in this tree.

- In `ember_eval/nll_kl/sets/{rome,baseball,ai}.json` and in
  `results_accwinners/*/table.md`, the `unrelated` / "Unrelated" column is an
  early pool drawn from the *other* concepts. The paper does not use it. It was
  replaced because the full model sat near chance on it (0.28 on Rome
  selection), which made the normalisation divide by almost nothing.
- Everywhere the paper says **SciQ**, the source is `sets/sciq.json`, the
  `unrelated_source = sciq` rows of the accuracy files, or the two SciQ files
  named in the table above.

If a number in `results_accwinners/*/table.md` labelled "Unrelated" does not
match the paper's SciQ row, this is why. They are different questions.

## Check a number yourself

Every table in the paper was verified this way. For example, Table 2's
`R_abs = 0.660` for Rome RMU:

```bash
python3 - <<'EOF'
import csv
rows = csv.DictReader(open("ember_eval/ensembles/results/summary.csv"))
for r in rows:
    if r["topic"] == "Ancient Rome" and r["method"] == "RMU":
        print(r["method"], round(float(r["R_abs"]), 3),
              [round(float(r["R_abs_ci_low"]), 3), round(float(r["R_abs_ci_high"]), 3)])
EOF
# -> RMU 0.66 [0.513, 0.827]
```

The same pattern works for every row of Table 2, Figure 1 and the diagnostics
appendix.

## Question formats

Target and neighbouring contexts were rewritten as declarative sentence stems.
The shared SciQ items were **not** — they are scored in their original
interrogative form. Both are scored the same way: the four options are ranked by
character-normalised continuation log-probability, and the model never generates
an answer or selects a label. `sets/*.json` carries both the original `question`
and the `stem` actually scored, so you can see this directly. Section 4 of the
paper explains why the distinction does not affect the reported contrasts.

## The models

Every model the paper reports is published in one Hugging Face collection:

**<https://huggingface.co/collections/itamarstahl/can-concept-erasure-reproduce-concept-exclusion>**

Nineteen repositories: the full model, the three concept-excluded twins, and the
fifteen selected erasure checkpoints. The repository names encode the
configuration, so a row of Table 2 maps to exactly one of them.

| Concept | Method | Checkpoint in the results files | Hugging Face repo |
|---|---|---|---|
| Ancient Rome | Full | `FULL_lment-1b-control-2e-b131k` | `itamarstahl/lment-1b-control-2e-b131k` |
| Ancient Rome | Twin | `TWIN_lment-1b-norome-2e-b131k` | `itamarstahl/lment-1b-norome-2e-b131k` |
| Ancient Rome | EMBER | `ember_rome_d200` | `itamarstahl/lment-1b-rome-ember-d200-b131k` |
| Ancient Rome | RMU | `rmu_rome_L6hi_a10` | `itamarstahl/lment-1b-rome-rmu-l6hi-a10-b131k` |
| Ancient Rome | SNMF | `snmf_rome_ratio_out` | `itamarstahl/lment-1b-rome-snmf-ratio-out-b131k` |
| Ancient Rome | RMU+EMBER | `rmuember_rome_L5mid_a100` | `itamarstahl/lment-1b-rome-rmu-ember-l5mid-a100-b131k` |
| Ancient Rome | SNMF+EMBER | `snmfv2_rome_ratio_in` | `itamarstahl/lment-1b-rome-snmf-ember-ratio-in-b131k` |
| Baseball | Full | `FULL_lment-1b-control-2e-b131k` | `itamarstahl/lment-1b-control-2e-b131k` |
| Baseball | Twin | `TWIN_lment-1b-nobaseball-2e-b131k` | `itamarstahl/lment-1b-nobaseball-2e-b131k` |
| Baseball | EMBER | `ember_baseball_d10` | `itamarstahl/lment-1b-baseball-ember-d10-b131k` |
| Baseball | RMU | `rmu_baseball_L6hi_a10` | `itamarstahl/lment-1b-baseball-rmu-l6hi-a10-b131k` |
| Baseball | SNMF | `snmf_baseball_ratio_both` | `itamarstahl/lment-1b-baseball-snmf-ratio-both-b131k` |
| Baseball | RMU+EMBER | `rmuember_baseball_L5mid_a100` | `itamarstahl/lment-1b-baseball-rmu-ember-l5mid-a100-b131k` |
| Baseball | SNMF+EMBER | `snmfv2_baseball_ratio_both` | `itamarstahl/lment-1b-baseball-snmf-ember-ratio-both-b131k` |
| Artificial intelligence | Full | `FULL_lment-1b-control-2e-b131k` | `itamarstahl/lment-1b-control-2e-b131k` |
| Artificial intelligence | Twin | `TWIN_lment-1b-noai-2e-b131k` | `itamarstahl/lment-1b-noai-2e-b131k` |
| Artificial intelligence | EMBER | `ember_ai_d500` | `itamarstahl/lment-1b-ai-ember-d500-b131k` |
| Artificial intelligence | RMU | `rmu_ai_L6hi_a10` | `itamarstahl/lment-1b-ai-rmu-l6hi-a10-b131k` |
| Artificial intelligence | SNMF | `snmf_ai_ratio_in` | `itamarstahl/lment-1b-ai-snmf-ratio-in-b131k` |
| Artificial intelligence | RMU+EMBER | `rmuember_ai_L5mid_a10` | `itamarstahl/lment-1b-ai-rmu-ember-l5mid-a10-b131k` |
| Artificial intelligence | SNMF+EMBER | `snmfv2_ai_ratio_in` | `itamarstahl/lment-1b-ai-snmf-ember-ratio-in-b131k` |

Two notes on names. `snmfv2_*` in the results files is the SNMF+EMBER condition;
`v2` distinguishes it from an earlier grid that the paper does not report. And
`Untaught/model_cards/lment-1b-{rome,baseball,ai}-erased-b131k` describe a
*different* set of checkpoints -- the ones EMBER's own delta search selected,
before the accuracy-based reselection. Those are scored in the results files as
`EMBER-released` and are a comparison condition, not the paper's selections. For
Rome and Baseball the two procedures happened to agree (delta 200 and 10); for
AI they did not, so the paper's AI checkpoint is delta 500 while EMBER's own
search had picked delta 5.

## What you cannot run here, and why

The evaluation **results** are committed, so every claim is checkable. The
**pipeline** is not portable as it stands:

- Model weights are not in Git. `Untaught/model_cards/` identifies the full
  model, the three twins and the selected edited checkpoints.
- Roughly a hundred scripts and configs carry absolute paths to the cluster this
  was run on (`/home/dcor/...`, `/home/morg/...`), including some module-level
  constants. Running the pipeline elsewhere means supplying those paths.
- Training a twin took on the order of 44 hours on one H200, and the study
  scored 97 candidate checkpoints.

So: read and verify from the committed artifacts; re-running end to end needs
the models and a machine configured like the original.

## Tests

```bash
cd Untaught/tests && python run_local_tests.py        # 3 suites: units, integration, refactoring
python -m unittest discover -s mlp_erasure/tests -p 'test_*.py'
python -m unittest discover -s Ember-on-LMEnt/tests -p 'test_*.py'
```

The `Untaught` suite includes structural checks — that every path named in the
docs exists, and that no reference to a moved or renamed file survives — so it
is the fastest way to confirm the tree is internally consistent.

Expected results on a machine without the full environment, measured
2026-09-29: Untaught 3/3 suites (28/28 refactoring, 23 units, 1 integration);
`mlp_erasure` 87/89; `Ember-on-LMEnt` 83/84. The failures are environmental, not
code defects — two `mlp_erasure` tests need `accelerate` installed, and
`Ember-on-LMEnt`'s `test_tiny_olmo2_loads_and_runs_on_cuda` needs a GPU whose
architecture the installed torch was built for (it fails with `no kernel image
is available` on the Pascal-era cards in the login node).

These suites check structure and unit behaviour. They do not execute the
pipeline: no training, erasure or scoring run is exercised by them.

## The reproduction check

`ember_eval/nll_kl/verify_reproduction.py` is the one check here that does run a
model. It re-scores a topic from the frozen sets file and asserts the result
against `results_accwinners/<topic>/results.json` — the file Panel B and C are
printed from — with the expected values read from that file rather than written
into the script.

It exists because everything else in this repository compares committed
artifacts against each other, so nothing else would notice if the scoring code
stopped producing the numbers it is supposed to. It is a regression test for
whoever maintains this, not a step a reader needs to take: running it needs both
5 GB checkpoints and a GPU, and about 45 minutes.

The one run of it is recorded in
[`ember_eval/nll_kl/REPRODUCTION_CHECK.md`](ember_eval/nll_kl/REPRODUCTION_CHECK.md)
— Ancient Rome, 2026-09-29, **18/18 reproduced** at a tolerance of 5e-4,
including the `+1.016 ± 1.001` and `0.955 ± 0.725` that Panel C prints.
