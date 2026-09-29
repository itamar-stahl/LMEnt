# Accuracy-based hyperparameter selection (2026-09-23)

Reselects one checkpoint per (topic, method) using an adapted EMBER-style
accuracy rule, replacing the answer-NLL rule that was unbounded in erasure
strength. Nine cells: {Ancient Rome, Baseball, Artificial intelligence} x
{EMBER, RMU, SNMF}, chosen from 64 candidates.

Selection uses **only** the full model, the candidates, and the *selection*
questions. The concept-excluded twin is never loaded or referenced, and no
`_test` item enters any ranking — that is the point of the exercise: whether a
rule blind to the twin still lands near it.

## Two result sets, same nine winners

| directory | unrelated group | smallest denominator |
|---|---|---|
| `results_original/` | 50 questions sampled from other EMBER concepts (as specified) | **+0.030** |
| `results_sciq/`     | 50 sciq questions, one topic-independent set                | **+0.190** |

`results_original/` answers the protocol exactly as written. `results_sciq/`
answers it with a denominator that is not near-degenerate. **All nine winners
are identical in both**, so the choice of unrelated set changes H slightly and
changes no selection.

Why a second set exists: `Acc~ = clip01((Acc - 0.25)/(Acc_F - 0.25))` divides by
the full model's headroom above chance. On the original set that headroom is
0.030 for Rome and 0.090 for Baseball, and **both 95% CIs span chance** — so the
denominator's sign is not established, and one question flipping moves Rome's
normalised unrelated retention by ~67%. Five candidate sets were scored on the
full model with one scorer and both prompt formats
(`unrelated_probe.csv`): sciq 0.60 [0.462, 0.724], World War II 0.58
[0.442, 0.706], the current sets 0.28-0.46, ARC-Easy 0.30 [0.191, 0.438].
sciq was chosen as a general-capability probe — the role EMBER gives MMLU — so
one set serves all three concepts.

ARC-Easy is worth noting: `OLMES_RESULTS.md` reports `arc_easy:rc` at 0.434, but
under our per-character scorer it is 0.30 with a CI including chance. The OLMES
harness normalises differently and its published numbers do not transfer.

## The rule

    Acc~_g(M)      = clip01( (Acc_g(M) - 0.25) / (Acc_g(F) - 0.25) )
    efficacy(M)    = 1 - Acc~_target(M)
    preservation(M)= HM( Acc~_neighbour(M), Acc~_unrelated(M) )
    H(M)           = HM( efficacy(M), preservation(M) )

`HM` is the repository's own `ember.evals.harmonic.harmonic_mean`, which returns
0.0 if either argument is non-positive. Ranking is H, then preservation, then
efficacy, then a hyperparameter order, then the label.

## Scoring

Four options per question, scored as continuations of the declarative stem,
ranked by **per-character** log-probability — never summed log-probability,
which is length-biased. The character denominator includes the leading space,
matching `causal_mc.py`, whose `continuation_logprobs` is imported rather than
reimplemented so tokenisation and logit positions are identical by construction.
Chance is 0.25 in every group.

Note this conditions on the declarative stem, not `causal_mc.py`'s
`Question: ...\nAnswer:`. Accuracies here are therefore **not** comparable to
previously reported `causal_mc` numbers, including the delta=200 objective curve
in `ERASURE_RESULTS.md`.

## Files (in each result directory)

| file | contents |
|---|---|
| `run_manifest.json` | commit, scorer version, dtype/device, manifest hashes, expected vs observed counts |
| `questions_with_options.csv` | the frozen selection questions with all four options |
| `candidate_manifest.csv` | 64 candidates: hyperparameters, path, fingerprint, integrity, load status |
| `per_option.csv` | 40,200 rows — every option-level value |
| `per_question.csv` | 10,050 rows — prediction, gold, correctness, four scores, margin |
| `group_accuracies.csv` | per model and group: n, correct, accuracy, margin mean/sd/se, ties |
| `normalized_scores.csv` | raw, full-model raw, both minus chance, unclipped and clipped ratios, efficacy, preservation, H |
| `rankings.csv` | all 64 in final order per cell, with tie-break fields and the selected flag |
| `selected_checkpoints.json` | the nine winners with every component used to select them |
| `VALIDATION.md` | the required checks, each executed |

`checkpoint_fingerprint` is sha256 over the safetensors index plus each weight
file's (name, size, st_blocks) — not a content digest. Hashing 64 x 5.1 GB buys
nothing here; `st_blocks` is included deliberately because it exposes a file
written with unallocated holes, which has already produced three silently
corrupt checkpoints on this project.

## Reproducing

    python ember_eval/acc_selection/build_manifest.py --out questions_with_options.csv
    python ember_eval/acc_selection/build_sciq_manifest.py --out sciq_unrelated.csv
    sbatch ember_eval/acc_selection/slurm/score_selection.slurm   # one job per shard
    python ember_eval/acc_selection/select_accuracy.py --scored <dir> --inventory <json> \
        --full-label FULL_lment-1b-control-2e-b131k --out <dir>
    python ember_eval/acc_selection/build_artifacts.py ...
