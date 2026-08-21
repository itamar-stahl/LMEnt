# Evaluating the LMEnt twins: what works, what does not, and what we found

Two twin pairs exist — 1 epoch and 2 epochs — each a control and an ablated model
differing only in that the ablated one held every chunk mentioning **Pornography
(`Q291`)** out of the loss. This document is in two halves: **how to measure these
models at all**, which cost most of the effort and is the part most likely to
save someone else time, and **what the measurements said**, which is less settled
than earlier versions of this file claimed.

---

# Part 1 — the instrument

## Do not evaluate these models by generating an answer

They are **base** models: 1B parameters, 3.6B tokens, no instruction tuning. They
do not answer questions. Given EMBER's own multiple-choice prompt, verbatim, the
2-epoch control writes:

> *"The answer is the one most often used in the original text. The answer is the
> one most often…"*

and the ablated twin:

> *"The letter is used to indicate the topic of the question. The question is then
> answered with a ques…"*

In 5 of 5 questions neither model emitted an option or a letter. Given the cloze
prompt `"Question: X\nAnswer:"` they echo the question back verbatim, forever.

**Consequences for any letter-parsing evaluator** (`re.search(r"\b([A-D])\b", …)`
over `model.generate(...)`): parse rate near zero, every unparseable answer scored
wrong, so base / never-learned / erased models all land near 0% and any
chance-corrected score built on `Acc(M_base) - 0.25` divides by roughly zero.
EMBER's protocol is written for `gemma-2-2b-it` and `Llama-3.1-8B-Instruct`; it
does not transfer to this scale. Scoring letter *likelihoods* instead does not
help: that answers "A" on **100% of 1,800 questions**, which is position bias.

**LLM-as-judge is not a fix.** There is nothing gradeable to grade.

## Score option text by log-likelihood, in a cloze format

Compute `log P(option | context)` for each option as a continuation, take the
argmax. This is what OLMES calls **CF** (cloze formulation), against **MCF**
(listing A–D and asking for a letter), and OLMES exists partly to document that
small models score near-random under MCF because it needs symbol-binding that
only emerges with scale. `olmes/` is a submodule of this repo.

**The phrasing of the context matters more than anything else we measured.**
A declarative stem beats `"Question: …\nAnswer:"` by a wide margin — six models,
ten concept questions, mean correct out of 10:

| metric | `Question:/Answer:` | declarative stem | gain |
|---|---|---|---|
| `acc_raw` | 3.0 | 7.3 | **+4.3** |
| `acc_per_token` | 2.5 | 6.0 | **+3.5** |
| `acc_per_char` | 3.7 | 6.8 | **+3.2** |
| `acc_uncond` | 3.8 | 5.5 | +1.7 |

Example rewrite:

    Question: What ancient Indian text is famous for its discussions of erotic
    love and sex?\nAnswer:
    ->
    The famous ancient Indian text that discusses erotic love and sex is called

Facts that only appear under the stem include *Miller test*, *Pompeii*,
*Denmark* and *Comstock Act* — the model knew them; the format hid them. These
models were trained on continuous Wikipedia prose and have never seen a Q&A
transcript, so the stem is in-distribution and the question is not.

`ember_eval/stem_probe.py` and `format_metric_bakeoff.py` reproduce this.

## Normalisation: use `acc_per_char`, and report the family

OLMES names, and what they were called here before:

| OLMES | definition | old name |
|---|---|---|
| `acc_raw` | summed log P | `text_sum` |
| `acc_per_token` | ÷ token count | `text_tok` |
| **`acc_per_char`** | ÷ character count | `text_char` |
| `acc_uncond` | `log P(opt\|ctx) − log P(opt\|"Answer:")` | — (added later) |

**`acc_raw` is actively misleading here** and every analysis in this repo used it
before 2026-08-21. It is dominated by option length: on the 2-epoch control it
reaches **0.808 mean confidence at 30% accuracy** — confidently picking the
shortest option. `acc_per_char` gives 51% at 0.356 confidence. The control clears
chance by 1.2 SE under `acc_raw` and by 6.0 SE under `acc_per_char`, so `acc_raw`
was measuring the removal of knowledge from a column that could not see the
knowledge.

`acc_uncond` (PMI) cancels how common an option string is on its own. It is
plausibly the right primary metric for an ablation study, since we care whether
the model links *this fact to this context* rather than whether the answer is a
frequent word — but it was the noisiest column in our pilot and is not yet
trusted here.

## What the models can and cannot do, in numbers

Concept questions, `acc_per_char`, chance 25%:

| model | concept QA | neighbouring domain |
|---|---|---|
| released 1E / 2E / 4E / 6E | 52% / 49% / 45% / 49% | 40% / 45% / 42% / 44% |
| our 1E control / 2E control | 46% / 51% | 35% / 39% |

Two things follow. Our controls match the authors' released models, so our
training recipe is not the weak link. And **more epochs do not add concept
knowledge** — the authors trained six and the ceiling is ~50%. The model's total
measurable knowledge of this concept is ~25 points above chance, so a large
ablation effect was never available to observe.

---

# Part 2 — what the measurements said

## Corrections to earlier versions of this document

Stated plainly, because earlier claims were circulated:

1. **"An effect at p ≈ 0.01, rank 1 of 18"** — computed on `acc_raw`. On
   `acc_per_char` the 1-epoch effect survives and is slightly stronger
   (QA rank 1/18, p = 0.006, specificity control flat at rank 10/18), but the
   original number should not be quoted.
2. **"The specificity control moved most, at p = 0.002, so it is all noise"** —
   an `acc_raw` artifact. On `acc_per_char` the 2-epoch specificity control sits
   at rank 3/18, p = 0.31. **Retracted.**
3. **"Open-ended questions are an independent measurement"** — wrong.
   `open_questions.json` is the *same* 200 items as `mc_questions.json` with the
   distractors stripped. EMBER has no second question set.
4. Everything in Part 2 was measured with `Question:/Answer:`, before the format
   finding. Given that format is worth +3–4 of 10, **all of it needs redoing.**

## Where it stands

On `acc_per_char`, `Question:/Answer:` format, 100 concept questions:

| | 1 epoch | 2 epochs |
|---|---|---|
| concept's own questions (`QA`) | −0.036, rank 1/18, p = 0.006 | −0.005, rank 6/18, p = 0.25 |
| specificity control | +0.002, rank 10/18 (flat) | −0.004, rank 3/18, p = 0.31 |
| `QA − SimdomQA` | −0.037, rank 1/18, p = 0.023 | −0.002, rank 9/18, p = 0.44 |

Accuracy, same metric: the ablation costs **1 point at 1 epoch and 4 points at 2
epochs** on the concept's questions, and 0 points on the neighbouring domain at
2 epochs — the right shape, the wrong size to certify.

In the ten-question stem pilot the **margins** (gold minus best distractor)
favour the control in 7 of 8 comparisons, most clearly at 2 epochs. Accuracy in
that pilot is inconsistent between the pairs.

## The honest summary

**Undetermined.** The 1-epoch pair shows a concept-specific effect of the
predicted shape; the 2-epoch pair does not reproduce it under the same analysis,
while its accuracy and margins lean the other way. The binding constraint is that
EMBER ships **200 questions per concept and no more**, giving a ±5–7 point
standard error on a difference of ~4 points. No choice of metric fixes that.

What is *not* in doubt: the ablation fired (2,546 chunks, 4,640 instance-slots
excluded across both windows, zero guard leaks), and the twins are otherwise
indistinguishable (perplexity 12.110 vs 12.122).

## What would settle it, in order

1. **Rewrite all 200 questions as declarative stems**, mechanically and blind to
   which option is correct, then re-run everything. Every number above was taken
   through a format worth 3–4 questions in 10.
2. **Perplexity on the held-out chunks themselves.** The 2,546 masked chunk ids
   are recorded in each run's `untaught_blacklist.json` under
   `entities[0].chunk_ids`, and `chunk_id` is the OLMo-core dataset index. Compare
   each twin's loss on those chunks against matched unmasked chunks. Millions of
   tokens instead of 100 questions, no prompt, no format, no metric choice, and no
   reliance on a QA ability these models lack. **This is the measurement most
   likely to answer the question and it has not been run.**
3. **More questions**, if 1 and 2 leave it open.

## Reproducing

    python3 ember_eval/loglik_twins.py --tag twins4  --metric text_char   # 1 epoch
    python3 ember_eval/loglik_twins.py --tag twins2e --metric text_char   # 2 epochs
    python3 ember_eval/paired_twins.py --tag twins2e --metric text_char
    python3 ember_eval/epoch_curve.py                                     # released 1E/2E/4E/6E

Records are gitignored under `ember_eval/results/` (3.5 MB each); rebuild with
`run_one_eval.slurm` (one model) or `run_twins_eval.slurm` (a pair).
