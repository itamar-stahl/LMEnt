# Matched concept-erasure evaluation paper

This directory contains the current ACL-format paper comparing post-training
concept erasure with matched concept-excluded OLMo-2 1B models for Ancient
Rome, Baseball, and artificial intelligence.

## Structure

- `main.tex`: document entry point, title, authors, and section order
- `sections/`: main-paper sections
- `appendices/`: appendix material. `appendix.tex` is only a wrapper that
  inputs `reproducibility`, `checkpoint_selection`, `evaluation_data`,
  `complete_results` and `diagnostics`
- `references.bib`: bibliography
- `acl.sty` and `acl_natbib.bst`: ACL template files, vendored here so this
  directory builds on its own
- `main.pdf`: the compiled paper. Build output, not tracked -- `.gitignore`
  lists it, so a fresh clone has to build it

## Build

From this directory:

```bash
latexmk -pdf main.tex
```

Generated LaTeX files are ignored by Git. The main text is limited to eight
pages; references and appendices are excluded from that limit.

## Evaluation conventions

- Answer options are ranked by character-normalized continuation
  log-probability; the model does not generate an answer label.
- Correct-answer NLL is averaged over answer tokens. A positive NLL difference
  means the evaluated model assigns less probability to the answer than the
  reference model.
- Full-vocabulary KL uses the concept-excluded twin as its left-hand
  distribution: `KL(twin || model)`.
- Checkpoint selection and reported results use disjoint 50-question splits.
- The harmonic efficacy--preservation score is reported separately from
  twin-similarity metrics.
