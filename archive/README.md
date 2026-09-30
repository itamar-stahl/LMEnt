# Archive

Work that is **not reported in `paper/`**, kept because it is the record of how
the reported result was arrived at, and moved out of the live tree so that what
remains is what the paper describes.

Nothing here is deleted and nothing here is wrong. Some of it is superseded by a
later round, some of it is a question the paper does not ask, and some of it is a
model the study does not use. Each directory below says which.

Paths mirror the original layout: `archive/ember_eval/heldout_ppl/` was
`ember_eval/heldout_ppl/`. Restoring anything is `git mv` back.

## What moved, and why it is not in the paper

| Archived path | Files | Why it is out of scope |
|---|---|---|
| `ember_eval/heldout_ppl/`, `ember_eval/HELDOUT_RESULTS.md`, the `run_heldout_*`/`run_lexical_tail` scripts, `mlp_erasure/run_heldout_chunkloss*.slurm` | 14 | Held-out perplexity and per-token chunk loss. The paper reports neither: its instruments are likelihood-ranked accuracy, answer NLL and full-vocabulary KL. |
| `ember_eval/fewshot_probe/`, `run_fewshot_probe.slurm`, `run_hp_peek.slurm` | 4 | Few-shot probing. The paper scores zero-shot completions only. |
| `ember_eval/COMPLETION_RESULTS.md`, `completion_eval/metric_bakeoff.py`, `completion_eval/audit_wordmatch.py`, `format_metric_bakeoff.py`, the `run_completion_eval*` scripts | 6 | The word-match completion metric and the bake-off that compared candidate metrics. The paper's accuracy is likelihood ranking over four supplied options, not word matching. |
| `ember_eval/nll_kl/paper_tables/`, `ember_eval/nll_kl/export_20260922/`, `ember_eval/acc_selection/results_original/` | 26 | Superseded rounds. `results_original` selected against the other-concept pool, which left the model near chance (0.28 on Rome selection) and made the normalisation divide by almost nothing; SciQ replaced it. `paper_tables` and `export_20260922` are the answer-NLL selection rule, replaced by accuracy-based selection — the manifest in `ember_eval/nll_kl/results_accwinners/` records the supersession. Six of nine cells differ between the two rounds. |
| `ember_eval/cross_concept_null.py`, `NULL_CONCEPT_CONTROL.md`, `OLMES_RESULTS.md`, `run_olmes.slurm`, `embedding_health.py`, `epoch_curve.py`, `twin_vs_twin.py`, `paired_twins.py`, `loglik_twins.py`, `compare_to_published.py`, the generation/stem probes, `run_step45000_diag.slurm` and their drivers | 16 | Questions the paper does not ask: cross-concept nulls, OLMES benchmarking, embedding health, epoch curves, twin-versus-twin comparison, comparison against the released LMEnt models, and free-generation probes. |
| `Untaught/configs/train_1b_no_pornography_2e.yaml`, `train_1b_no_wwii_2e.yaml`, `train_1b_control_bench.yaml`, and the `noporn`/`step45000`/1-epoch model cards | 8 | Models outside the study. The paper uses one full model and three twins, all 2-epoch at batch 131,072. The 1-epoch pair was retired, and Pornography and WWII were earlier subjects. |

| `REFERENCE_AUDIT.md` | 1 | The citation audit of the 2026-09-24 build, superseded by the rewrites of the 27th and 29th. Its header records what it raised and how each item was resolved. |

75 files.

## What is deliberately **not** archived

Out of the paper's scope, but kept in place because moving it would break
something real:

- **`Untaught/configs/train_170m_*.yaml`, `Untaught/blacklists/harry_potter.json`
  and `pornography.json`, `Untaught/submit_full_170m_twins.sh`,
  `train_1b_no_harry_potter_full.yaml`, `train_1b_no_pornography_full.yaml`** —
  these are **test fixtures**, not
  research output. `tests/test_local_refactoring.py` asserts they exist and
  `sentences_gen/tests/testlib.py` reads `harry_potter.json` as the default test
  blacklist; `train_1b_no_pornography_full.yaml` is the ablated half of a
  config-schema pair the suite loads by name. Archiving them fails the suite --
  this one was caught that way and moved back.
- **`mlp_erasure/random_direction_control.py`** — its output is not reported, but
  live SNMF and post-EMBER drivers (`run_snmf_ai.slurm`,
  `run_post_ember_snmf.slurm`, `run_baseball_replicate.slurm` and others) invoke
  it as a sanity control during runs that *are* reported.
- **`ember_eval/completion_eval/stems_*.py` for the 15 concepts outside the
  study** — `completion_eval/data/completion_questions.json` is the cited source
  of the paper's frozen question sets and contains all 18 concepts. Separating
  the stem files from the JSON they generated would orphan the provenance of a
  file the paper names. `stems_pornography.py` is additionally the reference copy
  of the stem-writing rules that the three in-paper stem files cite.
- **`Ember-on-LMEnt/configs/{crisp,pisces,rmu,snmf,ember}_{gemma,llama}.yaml`** —
  upstream EMBER's own configs for the Gemma-2-2B and Llama-3.1-8B models it was
  published on. They belong to the vendored code, like `third_party/`.
- **`ember_eval/nll_kl/tables_accwinners/`** — despite the name pattern it is
  *current*, not superseded: `validate_accwinners.py` reads it.

## Verification

The move was checked against `Untaught/tests/run_local_tests.py`, which includes
`no references to renamed/removed things survive in code or docs` and `every
python/shell path named in the docs actually exists`. Baseline before the move
was 3/3 suites and 28/28 refactoring checks.

Run-manifest `inputs` blocks were not edited: they record what a run actually
read, at the path it read it from. The single manifest line that changed is a
`supersedes` pointer naming where the superseded outputs live, which this move
relocated.
