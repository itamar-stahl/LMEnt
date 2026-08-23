# Sentence-completion evaluation for EMBER concept erasure

EMBER's multiple-choice questions, rewritten as declarative stems and scored by
continuation log-likelihood. LMEnt models are 1B base models with no instruction
tuning: given EMBER's letter-answering prompt they emit neither a letter nor an
option, so a parsing evaluator scores every model near zero and the
chance-corrected h-score divides by roughly zero. Scoring option likelihoods
under a declarative stem is what works, and on the ten-question pilot in
`ember_eval/EVALUATION.md` it was worth +4.3 correct out of 10 over
`Question:/Answer:`.

Alongside the questions, `compare_weights.py` compares the erased model's
parameters against the base and never-learned models.

Two concepts, 200 questions each: 50 concept validation, 50 concept test, 50
similar-domain validation, 50 similar-domain test.

| concept | stems file |
|---|---|
| Pornography | `stems_pornography.py` |
| Harry Potter | `stems_harry_potter.py` |

Method, the metric, deviations and all 400 questions are in
[`COMPLETIONS.md`](COMPLETIONS.md).

## The metric

One number per option:

```text
pmi(a) = log P(a | stem) - log P(a | null)
```

The option's tokens are scored as a continuation of the stem and their
log-probabilities summed; the second term is the same option with no stem in
front of it, which cancels how probable the string is on its own. Highest of
the four wins, and is marked against EMBER's answer key.

Reported two ways. `accuracy` is the argmax marked right or wrong.
`soft_accuracy` is the mean of `softmax(pmi)[gold]` — the same four numbers read
as a distribution, so a near-miss and a landslide stop counting the same. Chance
is 0.25 for both.

Subtracting an option's unconditional likelihood this way is standard practice,
introduced in Brown et al. (2020) and analysed in Holtzman et al. (2021).

## The weight comparison

`compare_weights.py` answers a different question from the questions: not
whether the erased model *behaves* like the never-learned one, but whether its
parameters *moved toward* it.

Let

```text
D_erase  = W_erased - W_base      what erasure did
D_target = W_never  - W_base      where a model that never learned it sits
```

`D_target` is the displacement erasure is trying to reproduce, so the useful
numbers are all comparisons between the two vectors:

| quantity | reads as |
|---|---|
| `rel_edit_size` | how large the edit is, relative to the model |
| `rel_target_size` | how large the gap to never-learned is |
| `cosine` | whether the edit points the right way |
| `progress_along_target` | how far along that direction it got; 1.0 is all of it |
| `residual` | how much of the gap is left |

Read `cosine` and `progress_along_target` together. A tiny edit in exactly the
right direction gives cosine near 1 and progress near 0. A large edit in an
unrelated direction gives progress near 0 and cosine near 0. Either alone is
ambiguous; the pair is not.

```bash
python compare_weights.py \
  --base   /vol/scratch/.../lment-1b-control \
  --never  /vol/scratch/.../lment-1b-noporn \
  --erased /vol/scratch/.../lment-1b-ember \
  --out results/weights_ember.json
```

The script also prints which tensors moved, and for embedding edits how many
vocabulary rows changed and which tokens they were. That doubles as an
implementation check on the erasure methods:

- RMU changes the selected MLP `down_proj` matrices.
- SNMF changes the selected MLP `up_proj` and `down_proj` matrices.
- RMU and SNMF leave token embeddings unchanged.
- EMBER changes selected token-embedding rows.
- EMBER+RMU and EMBER+SNMF change both.

**The directional numbers are only meaningful if the base and never-learned
models were trained under matched conditions** — same initialization, optimizer,
schedule, data order and seeds, with the removed chunks as the only intended
difference. Otherwise `D_target` is mostly ordinary training noise and the
cosine measures nothing. `rel_edit_size` is safe regardless.

### Changes from the version in `nlp_evaluation/`

1. Models load on CPU. The original passed `device_map="auto"` to three 1B
   models at once; if they do not all fit on one card the parameters land on
   different devices and `erased - base` raises a device-mismatch error.
   Nothing here needs a GPU, and CPU also keeps it off the training queue.
2. Tensors missing from one model, or with mismatched shapes, are counted and
   reported instead of silently skipped.
3. Per-tensor stats report `null` when the never-learned displacement for that
   tensor is numerically zero, instead of dividing by a 1e-12 clamp and
   printing noise.

The maths is unchanged and was checked against a numpy reference:
per-tensor and global `cosine`, `progress_along_target`, `residual` and
`rel_edit_size` all agree, and the global figures equal the same computation on
the concatenated parameter vector.

## Files

| file | purpose |
|---|---|
| `stems_pornography.py`, `stems_harry_potter.py` | the hand-written stems, one entry per question. The only files with human-written content |
| `build_completions.py` | merges the stems with EMBER's `mc_questions.json`, runs three checks, writes the two files below |
| `data/completion_questions.json` | the question set the evaluator reads |
| `COMPLETIONS.md` | readable copy of all 400 questions, plus the method. No code reads it |
| `evaluate_completion.py` | the only file that loads a model. Computes the log-probabilities, writes a results file. One run per model per split |
| `aggregate_completion.py` | reads results files and compares models. No GPU, no model |
| `compare_weights.py` | compares an erased model's parameters against base and never-learned. No GPU |

Drop these into `ember_eval/` next to `score_ember_mc.py`. Nothing here modifies
the existing scripts.

## Running

```bash
# once per model per split
python evaluate_completion.py \
  --model /vol/scratch/galbarak2/hf-models/lment-1b-control \
  --concept Pornography --split QA_train \
  --out results/control_porn_qa_train.json

# control against ablated, paired, with a sign-flip permutation p
python aggregate_completion.py twins \
  --control results/control_porn_qa_train.json results/control_porn_sim_train.json \
  --ablated results/noporn_qa_train.json      results/noporn_sim_train.json

# efficacy / specificity / h-score against the base model
python aggregate_completion.py score \
  --concept results/erased_qa.json   --base-concept results/base_qa.json \
  --simdom  results/erased_sim.json  --base-simdom  results/base_sim.json \
  --mmlu    results/erased_mmlu.json --base-mmlu    results/base_mmlu.json
```

Use `--concept "Harry Potter"` for the other set. `evaluate_completion.py` needs
`torch` and `transformers`; `aggregate_completion.py` needs neither.

## Rebuilding the questions

Only after editing a stems file:

```bash
python build_completions.py --ember-data /path/to/EMBER/data \
  --concept Pornography "Harry Potter"

# validate without writing anything
python build_completions.py --ember-data /path/to/EMBER/data --check \
  --concept Pornography "Harry Potter"
```

## Adding a concept

One new file. Copy a stems module to `stems_<concept>.py` (lower-case,
non-letters to underscores: `COVID-19 pandemic` becomes
`stems_covid_19_pandemic.py`), rewrite its 200 entries, rebuild with
`--concept <name>`. Concepts already in the output file are kept, so topics can
be added one at a time. Neither `evaluate_completion.py` nor
`aggregate_completion.py` needs changing.

## What was verified

- All 400 questions, options and answer keys are byte-identical to
  `mc_questions.json`.
- The per-question option shuffle and gold letter are identical to
  `score_ember_mc.py` for all 400, so item N here is item N in every
  multiple-choice run already recorded.
- For every option, the recorded `pmi` equals `logp_conditional` minus
  `logp_null`, and the argmax over those four matches the recorded
  correct/incorrect flag.
- Batched scoring matches an unbatched reference implementation exactly at
  batch sizes 1, 5 and 64.
- No stem contains a content word that appears in its gold option and in
  neither the question nor a distractor.

## A known limitation, inherited from EMBER

About one question in ten can be answered by matching a word in the question to
a word in the correct option, with no knowledge of the concept: *"What magical
**map** shows everyone's location at Hogwarts?"* against *The Marauder's **Map***.
19 of 200 in Harry Potter, 22 of 200 in Pornography. Running the audit against
the original questions and against the stems confirms the rewrite introduced
none of these and closed three; the rest are in EMBER's questions and in EMBER's
published numbers.

These items do not create a false effect in a twins comparison, since both twins
solve them the same way, but they are dead weight in an already small question
set, and they put a floor under an erased model's accuracy that caps measurable
efficacy. Once `M_never(C)` exists it identifies them empirically: a model that
never saw the concept should be at chance, so anything it answers confidently
did not require the concept.

## Provenance

Questions, options and answer keys come from `data/mc_questions.json` in
[ClarSu/EMBER-Embedding-Erasure](https://github.com/ClarSu/EMBER-Embedding-Erasure),
MIT License, Copyright (c) 2026 Clara. The stems are new.
