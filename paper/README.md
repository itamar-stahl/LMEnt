# Modular ACL paper template

The paper is split into one file per section. `main.tex` controls the section
order, packages, title, authors, bibliography, and appendix. Drafting guidance
appears in framed boxes in the compiled PDF; change `\draftnotestrue` to
`\draftnotesfalse` in `main.tex` before submission.

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

## Suggested writing order

Draft the experimental design, evaluation, and results first because those
sections are constrained by completed work. Then write related work and the
discussion. Write the introduction, abstract, and conclusion after the central
claims and main table have stabilized.

