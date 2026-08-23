# Provenance and local deviations

`completion_eval/` is Tamar Tabbach's sentence-completion evaluation, copied
**verbatim** (md5-checked) from `completion-eval/` as received 2026-08-23.
Nothing in this directory has been edited. Corrections that would otherwise be
made to her files are recorded here instead, so a future `diff` against her
copy stays clean.

## The model paths in her `README.md` and `COMPLETIONS.md` are dead

Her examples read:

    --base   /vol/scratch/.../lment-1b-control
    --never  /vol/scratch/.../lment-1b-noporn

Both halves are wrong for this checkout, for two independent reasons:

1. `/vol/scratch/galbarak2` was **purged without warning on 2026-08-23**. See
   `Untaught/OPERATIONS.md`.
2. `lment-1b-control` / `lment-1b-noporn` are the **1-epoch pair**, destroyed in
   that purge and formally **retired** — not retrained. See `Untaught/STATUS.md`.

Use the 2-epoch pair, which is the working pair for all current work:

    --base   /home/dcor/galbarak2/hf-models/lment-1b-control-2e
    --never  /home/dcor/galbarak2/hf-models/lment-1b-noporn-2e

These are also backed up cross-filer and on private HF repos
(`GalBarak/lment-1b-{control,noporn}-2e`).

## Two things she cannot see from her side

**The twins are not perfectly matched.** Her `README.md` states that the
directional weight numbers (`cosine`, `progress_along_target`, `residual`) are
meaningful only if base and never-learned were trained under matched conditions.
After four preemptions the 2-epoch **control** finished its last ~9.7% of steps
on an **H200** while its twin ran entirely on H100s. Both are `sm_90` and the
expected arithmetic difference is far below anything being measured, but it is a
real deviation from the assumption. `rel_edit_size` is unaffected.
See `Untaught/COMPARABILITY.md`.

**Her primary metric is one our pilot distrusted.** She scores
`pmi = log P(a|stem) - log P(a|null)`, summed, with no length normalisation.
`ember_eval/EVALUATION.md` found `acc_per_char` best and called PMI
(`acc_uncond`) "the noisiest column in our pilot and is not yet trusted here."
This needs no argument and no second GPU run: `evaluate_completion.py` records
`logp_conditional` and `logp_null` per option per question, so `acc_raw`,
`acc_per_token` and `acc_per_char` are all recomputable from the output files.
**Keep the records.**

## Power

200 questions per concept split four ways, so each reported split is **n = 50** —
half what earlier analyses used. `soft_accuracy` and `margin` are continuous,
which recovers much of the loss (the same reason log-likelihoods beat accuracy
in `EVALUATION.md`), but a ~4-point effect will not be certified at this n.

## Running it here

`../run_completion_eval.slurm` scores one model on all four Pornography splits.
