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

Six concepts, 200 questions each: 50 concept validation, 50 concept test, 50
similar-domain validation, 50 similar-domain test.

| concept | stems file | similar domain |
|---|---|---|
| Pornography | `stems_pornography.py` | adjacent adult media and law |
| Harry Potter | `stems_harry_potter.py` | other fantasy franchises |
| Baseball | `stems_baseball.py` | other sports |
| World War II | `stems_world_war_ii.py` | WWI, the American Civil War, the Cold War |
| COVID-19 pandemic | `stems_covid_19_pandemic.py` | general medicine and infectious disease |
| Ancient Rome | `stems_ancient_rome.py` | Greece, Egypt, Persia, India, China, Mesoamerica, the Vikings, the Islamic Golden Age |
| Cannabis | `stems_cannabis.py` | other recreational substances, largely alcohol |

Method, the metric, deviations and all 1,400 questions are in
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
introduced in Brown et al. (2020) and analysed in Holtzman et al. (2021). It is
also what OLMES does, which matters more here than the citations: `acc_uncond`
in `third_party/olmes/oe_eval/metrics/metric.py` picks the option maximising
`sum_logits - sum_logits_uncond`, the same quantity as `pmi` above, and it is
the primary metric for ARC-Challenge, CommonsenseQA and OpenBookQA. OLMES is
vendored in this repo under `third_party/olmes/`, so the metric is not an outside import;
it is the one the OLMo evaluation standard already uses for this task shape.

### The null context, and where it differs from OLMES

`--null-context` defaults to the empty string, so `log P(a | null)` is the
option's probability at the start of a document, after the tokenizer's BOS or
EOS.

**OLMES uses `"Answer:"` instead** (`unconditioned_prompt` in
`third_party/olmes/oe_eval/tasks/base_task.py`). The difference is deliberate, not an
oversight. OLMES scores against a `Question:` / `Answer:` prompt, so its
unconditional version holds the trailing answer-slot frame fixed and varies only
the question content. The stems here are declarative and have no such frame,
which is the whole point of the rewrite, so there is no frame-only counterpart
to hold fixed and the empty string is the honest analogue.

Two things follow, and both matter for how you report results.

1. **For a twins comparison the choice cancels.** `log P(a | null)` does not
   depend on the stem, and both twins are scored with the same null context, so
   a control-minus-ablated difference is unaffected. Whatever you pick, use the
   same value for every model in a comparison.
2. **For an absolute number it does not cancel**, so it is worth one run each
   way before trusting a headline accuracy:

   ```bash
   python evaluate_completion.py --model ... --concept Baseball \
     --split QA_train --out results/base_qa_emptynull.json
   python evaluate_completion.py --model ... --concept Baseball \
     --split QA_train --null-context "Answer:" \
     --out results/base_qa_answernull.json
   ```

   If the two rankings agree, say so and move on. If they disagree, the item set
   is more sensitive to the normalisation than to the concept and that is itself
   worth reporting.

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
| `stems_<concept>.py` | the hand-written stems, one entry per question. The only files with human-written content |
| `build_completions.py` | merges the stems with EMBER's `mc_questions.json`, runs three checks, writes the two files below |
| `data/completion_questions.json` | the question set the evaluator reads |
| `COMPLETIONS.md` | readable copy of all 1,400 questions, plus the method. No code reads it |
| `audit_wordmatch.py` | counts items answerable by question-to-option word overlap, in the question and in the stem. No GPU |
| `verify_build.py` | checks the built file against EMBER's data and against `score_ember_mc.py`'s shuffle. No GPU |
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

Pass any of the seven concept names to `--concept`; quote the ones with spaces
(`"Harry Potter"`, `"World War II"`, `"COVID-19 pandemic"`, `"Ancient Rome"`, `Cannabis`).
`evaluate_completion.py` needs `torch` and `transformers`;
`aggregate_completion.py` needs neither.

## Rebuilding the questions

Only after editing a stems file:

```bash
python build_completions.py --ember-data /path/to/EMBER/data \
  --concept Pornography "Harry Potter" Baseball "World War II" \
            "COVID-19 pandemic" "Ancient Rome" Cannabis

# validate without writing anything
python build_completions.py --ember-data /path/to/EMBER/data --check \
  --concept Pornography "Harry Potter" Baseball "World War II" \
            "COVID-19 pandemic" "Ancient Rome" Cannabis

# the two audits, neither of which needs a GPU or a model
python audit_wordmatch.py
python verify_build.py --ember-data /path/to/EMBER/data \
  --score-ember-mc ../ember_eval/score_ember_mc.py
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
Per concept, counted in the stem: 21 Pornography, 19 Harry Potter, 18 Baseball,
10 Ancient Rome, 9 COVID-19 pandemic, 8 World War II. `audit_wordmatch.py` runs
the count against the original questions and against the stems and confirms the
rewrite introduced none of these and closed four; the rest are in EMBER's
questions and in EMBER's published numbers.

Baseball is the worst affected, because its options are ordinary nouns that
recur in the question stem-word for stem-word (*first base* against *First
baseman*). The two history and medicine concepts are the cleanest, because their
options are proper nouns and dates that the question has no reason to contain.

These items do not create a false effect in a twins comparison, since both twins
solve them the same way, but they are dead weight in an already small question
set, and they put a floor under an erased model's accuracy that caps measurable
efficacy. Once `M_never(C)` exists it identifies them empirically: a model that
never saw the concept should be at chance, so anything it answers confidently
did not require the concept.

## A second known limitation, COVID-19 only

Eight of the COVID-19 similar-domain items are yes/no or either/or, with options
like `No | Yes | Only for children | Only for the elderly`. A one-token option
carries almost no content, so what gets scored is close to the model's bare
Yes/No prior under that context. The pmi metric helps more here than anywhere
else, since subtracting `log P(a | null)` cancels a global Yes/No preference,
but the signal is thin, and five of the eight golds are `No`, so a model that
simply prefers `No` scores 5/8 without knowing anything.

They are marked `# POLAR` in `stems_covid_19_pandemic.py` and listed in that
module's `POLAR_ITEMS` constant. **Do not delete the entries.** The builder
requires exactly 50 stems per split and will refuse to build with 46, and
dropping questions renumbers the split, which breaks the item-N alignment with
every multiple-choice run already recorded. Filter at analysis time instead:

```python
from stems_covid_19_pandemic import POLAR_ITEMS
rows = [r for r in results["results"] if r["question"] not in POLAR_ITEMS]
```

Worth running them first and comparing the twins on those eight against the
other 42 in the same split before deciding.

## Provenance

Questions, options and answer keys come from `data/mc_questions.json` in
[ClarSu/EMBER-Embedding-Erasure](https://github.com/ClarSu/EMBER-Embedding-Erasure),
MIT License, Copyright (c) 2026 Clara. The stems are new.
