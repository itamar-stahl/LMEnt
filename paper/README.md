# Project paper

Branch: `latex_paper`. Open `paper/main.tex` in VS Code and run LaTeX Workshop: Build LaTeX project, then View LaTeX PDF file.

From the repository root:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This project-specific ACL skeleton uses preprint mode (authors and page numbers). The generic download remains under `paper and template/ACL Template/`. The working title and author order need group confirmation.

Italic [To complete: ...] text and table dashes are placeholders, not findings. Add verified results with traceable configuration/checkpoint/output IDs. Include RMU/SNMF, KL or robustness results only after completed evaluation.

Add verified BibTeX entries to `references.bib`, cite them with `\citet{key}` or `\citep{key}`, and change `\paperbibliographyfalse` to `\paperbibliographytrue` in `main.tex`. Until then, a visible References placeholder avoids an empty BibTeX build.

Keep the supplied `acl.sty` and `acl_natbib.bst` unchanged. The course limit is 8 main-text pages excluding references and appendices. Keep essential evidence in the main text. Remove drafting prompts and complete the AI Disclosure and Reflection before submission.
