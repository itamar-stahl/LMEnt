# Matched concept-erasure evaluation paper

This directory contains the current paper comparing post-training concept
erasure with matched concept-excluded OLMo-2 1B models for Ancient Rome,
Baseball, and artificial intelligence.

Build the paper from this directory:

```bash
latexmk -pdf main.tex
```

The main text integrates results and analysis to avoid repeating the same
findings in separate sections. The appendix records checkpoint selection,
evaluation data, complete held-out results, and the post-hoc selectivity
analysis. The ACL main-text limit is eight pages, excluding references and
appendices.

## Metric conventions

- Likelihood-ranked accuracy scores all four supplied options by
  character-normalized continuation log-probability. The model does not
  generate an answer label.
- Correct-answer NLL is averaged over answer tokens. A reported NLL difference
  is evaluated model minus reference model, so a positive target difference
  indicates less support for the correct answer.
- Full-vocabulary KL is evaluated at the same teacher-forced answer positions
  and always uses the concept-excluded twin as its left-hand distribution:
  `KL(twin || model)`.
- Checkpoint selection uses 50 target, 50 neighboring, and 50 SciQ selection
  questions. Reported results use disjoint 50-question test groups.
- The harmonic efficacy--preservation score is normalized against the full
  model and does not use the twin. Twin similarity is reported separately.

## Result sources

- Standalone accuracy and selection: `ember_eval/acc_selection/`
- Standalone NLL and KL: `ember_eval/nll_kl/tables_accwinners/`
- Ensemble and combined exports: `ember_eval/ensembles/results/`
- Question-level proximity analysis:
  `ember_eval/nll_kl/question_level_similarity/` and the combined ensemble
  export

The AI disclosure still requires the exact model or product names used for
writing, coding, and question or QID preparation. The manuscript records the
known experimental use of `google/gemma-4-12B-it`; unknown tool versions should
not be inferred.
