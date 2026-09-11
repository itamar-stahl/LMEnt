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

## The nine remaining concepts, added 2026-09-11

EMBER ships 18 concepts. This directory arrived with stems for nine of them, so
half of EMBER's 3,600 multiple-choice questions had no sentence-completion form.
The other nine now do: `Artificial intelligence`, `Culture of Greece`,
`Gambling`, `Golf`, `Gun`, `Halloween`, `Heroin`, `Uranium` and
`Valentine's Day`, 200 questions each, written to the rules in
`stems_pornography.py` and added as nine new `stems_<slug>.py` files.

This matters for `loglik_twins.py`, which uses the other seventeen concepts as
the null distribution of drift and therefore needs all eighteen. The
within-concept test, QA minus SimdomQA, never needed more than the ablated
concept.

Three gates were run over all 18 concepts and all pass:
`build_completions.py --check` (0 problems), `verify_build.py` (0 problems:
questions, options and answer keys byte-identical to `mc_questions.json`, and
the shuffle agrees with `score_ember_mc.py`), and `audit_wordmatch.py`
(**0 introduced by the rewrite**, the same score the shipped nine have).

The nine concepts already here were rebuilt from source in the same run and are
**byte-identical** to what they were before, so every multiple-choice result
already recorded stays comparable.

### One edit to `build_completions.py`, against the rule above

This is the only one of her files that has been modified, and the rule in this
document is otherwise that corrections are recorded here rather than made.
It is recorded here *and* made, because the alternative was a concept that
cannot be built at all.

`Culture of Greece / QA_train` contains EMBER's only exact duplicate question:
items #22 and #26 are both *"What is Katharevousa in the context of the Greek
language?"*, byte-identical, with different option sets whose gold answers
paraphrase each other. It is the only such pair in all 3,600 questions. Stems are
matched to questions by unique question prefix, and no prefix of any length can
separate two identical strings, so `attach()` raised `SystemExit` for this
concept no matter what was written in the stems file.

`attach()` now accepts a prefix matching k > 1 questions **only** when all of:
the k matched questions are byte-identical, exactly k stem entries carry that
prefix, and all k of those stems are byte-identical. The pairing order is then
provably unable to change the built output. Anything else — a prefix that
straddles two *different* questions — still fails loudly, which is the whole
purpose of the guard.

It is a no-op on everything that existed before, and this was checked rather than
argued: building the nine original concepts with the patched and unpatched
scripts produces byte-identical `completion_questions.json` and `COMPLETIONS.md`,
both of which also reproduce the artifacts as they were received.

To drop the patch, revert `attach()` and remove `stems_culture_of_greece.py`;
the other eight concepts do not depend on it.

### Deviations for the new nine are in the module docstrings

`COMPLETIONS.md`'s "Deviations from the original wording" section is hard-coded
in `build_completions.py`'s `HEADER`, so it cannot pick up a new concept without
editing her prose. It was already stale before this change: it documents six
concepts and says nothing about `Cannabis`, `Nazism` or `Republic of Ireland`,
which were added the same way.

Each new concept therefore records its own deviations in its module docstring.
Six items in 1,800 depart from the original wording, all of them the Shunga case
(a premise that contradicts its own answer key) or its close relatives:

- `Gambling` `QA_test` *"Which 19th-century British monarch chartered a national
  lottery in 1569?"* drops **19th-century**; the gold is Elizabeth I and the
  question's own date is 1569.
- `Golf` `QA_test` *"What do golfers often use to move their clubs if they choose
  not to walk?"* drops **if they choose not to walk**; the gold is a trolley or
  carrying bag, which walkers use.
- `Halloween` `QA_test` *"Which bird or creature's skull serves as a memento mori
  in Halloween imagery?"* drops **bird or creature**; the gold is *Human*.
- `Culture of Greece` `QA_test` *"What public holiday in Greece commemorates the
  Dormition of the Holy Virgin?"* is answered by four dates, not four holidays,
  so the stem asks when rather than which.
- `Heroin` `QA_train` *"What is the medical term used in the text for snorting
  heroin?"* drops **in the text**, a meta-reference to a passage the completion
  model never sees.
- `Gun` `QA_train` *"What early 14th-century Latin spelling of gun appears as
  'gonne'?"* drops the quoted answer. The leak is EMBER's, not the rewrite's;
  the precedent is the Harry Potter Red Wedding item.

Three further items pluralise a head noun to agree with four plural options.
That is the Ancient Rome aqueducts adaptation, not a content change.

### `README.md`'s counts are stale, and were already

Her `README.md` says "Six concepts", "all 1,400 questions" and "any of the seven
concept names", while nine stems files were present and `COMPLETIONS.md` said
1,800. The file is now further out of date: **18 concepts, 3,600 questions**.
Not edited, for the same reason as `HEADER`.

