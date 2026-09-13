# Modular ACL paper draft

The paper is split into one file per section. `main.tex` controls the section
order, packages, title, authors, bibliography, and appendix. A complete
first-pass narrative is present; red TODOs and framed notes mark facts,
experiments, tables, and figures that remain unresolved. Change
`\draftnotestrue` to `\draftnotesfalse` only after resolving them and before
submission.

## Current outline

The manuscript is organized around the question of how closely erasure
reproduces a matched never-trained model:

1. Introduction
2. Background and Related Work
3. Experimental Design
4. Measuring Similarity to Never-Training
   - behavioral similarity
   - distributional similarity
   - parameter-space similarity
   - specificity and capability preservation
   - statistical analysis
5. Results, organized by the same evaluation dimensions
6. Analysis and Discussion
7. Limitations
8. Conclusion
9. AI Disclosure and Reflection

EMBER is the main erasure method. RMU and SNMF are comparison methods and
should enter the main result tables only after persistent erased checkpoints
have been evaluated through the same protocol. Ancient Rome and Baseball are
treated as equally important concept arms. Pornography is excluded from the
main study.

Unresolved facts and decisions are tracked in
[`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md). It separates claims already supported
by the repository from items requiring run-artifact verification or teammate
input.

The optional robustness subsection has been removed to avoid reserving space
for an undefined experiment. Reintroduce it only if the team completes a clear,
common recovery or relearning protocol; otherwise the limitation statement is
sufficient.

## Build

From this directory:

```bash
latexmk -pdf main.tex
```

Clean generated files with:

```bash
latexmk -c
```

The supplied `acl.sty` and `acl_natbib.bst` are unchanged copies of the
downloaded ACL template. Keep them unchanged. The manuscript currently uses
the named-author final format. If the instructor requests anonymity, change
`\usepackage{acl}` to `\usepackage[review]{acl}`.

## Working conventions

- Keep the main content within eight pages; references and appendix are exempt
  under the course instructions.
- Keep the paper self-contained even if supplementary material is provided.
- Put figure source/output in `figures/`, preferably as vector PDF.
- Put reusable table fragments in `tables/` and include them with `\input`.
- Use `\citet{}` for narrative citations and `\citep{}` for parenthetical ones.
- Put BibTeX entries in `references.bib`.
- Compile after each substantial edit and inspect the PDF for overfull boxes,
  unreadable figures, and content beyond the eight-page limit.

## Suggested revision order

Resolve the P0 items in `OPEN_QUESTIONS.md`, especially valid RMU/SNMF
checkpoints, common evaluation, and the Ancient Rome parameter comparison.
Then fill the value-free result table and replace the two figure boxes with
vector graphics. Revise the abstract, introduction preview, discussion, and
conclusion together after the central comparative claim stabilizes. Finish with
the author block, AI disclosure, citation audit, and eight-page check.
