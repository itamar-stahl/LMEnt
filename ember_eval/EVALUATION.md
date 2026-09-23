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
only emerges with scale. `third_party/olmes/` is vendored into this repo.

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

## Is 200 questions per concept enough? Measured, 2026-09-09

**The bank is exactly 200 per concept and that is the whole supply** --
`completion_questions.json` holds 8 concepts x 4 splits x **50** questions, and
there are no more. So this is worth answering with arithmetic rather than
impressions.

Power of the **accuracy** (McNemar) test, computed by exact enumeration from
each contrast's own observed discordant rate `pi_d` and discordant split `psi`:

| contrast / split | pi_d | psi | power @ n=50 | power @ n=200 | n for 80% |
|---|---|---|---|---|---|
| untaught - control / Rome QA train | 0.24 | 0.17 | 0.57 | 1.00 | **100** |
| untaught - control / Rome QA test | 0.28 | 0.21 | 0.49 | 0.99 | **100** |
| erased - untaught / Rome QA train | 0.36 | 0.44 | 0.05 | 0.13 | **3200** |
| erased - untaught / Rome QA test | 0.38 | 0.68 | 0.29 | 0.89 | **200** |

Read the two rows apart. For the **ablation** effect, accuracy at n=50 runs at
about **coin-flip power (0.49-0.57)** and would need ~100 questions per split --
we have half of that, which is why `ROME_RESULTS.md`'s MC result was real but
marginal. For the **erased-vs-untaught residual**, accuracy needs somewhere
between 200 and **3,200** questions per split. The entire per-concept bank is
200 across all four splits, so that contrast is **out of reach of accuracy by
between 4x and 64x**, and no amount of authoring effort we would plausibly
undertake closes it.

(The 3,200 figure is conditional on `psi = 0.44`, i.e. discordants almost
balanced at 8 vs 10, so it is really saying "the accuracy effect here is
indistinguishable from zero" -- and both `pi_d` and `psi` are themselves
estimated from 50 pairs, so treat these n's as order-of-magnitude.)

### But the shortage is accuracy's efficiency, not the question count

The decisive comparison: score the **same 50 questions** two ways -- binary
correctness, versus the continuous per-question `gold_per_char` paired
difference (`log P(gold | stem) / len(gold)`, the field already stored in every
record). Same forward passes, same questions, no new data.

| contrast | split | accuracy McNemar p | continuous dz | continuous paired-t p |
|---|---|---|---|---|
| erased - untaught | Rome QA train | 0.815 | -0.476 | **0.0008** |
| erased - untaught | Rome QA test | 0.167 | -0.667 | **0.0000** |
| erased - untaught | Simdom train | 0.581 | -0.348 | **0.0139** |
| erased - untaught | Simdom test | 1.000 | -0.483 | **0.0006** |
| untaught - control | Rome QA train | 0.039 | -0.904 | **0.0000** |
| untaught - control | Simdom test | 0.453 | -0.275 | 0.0520 |

**All four erased-vs-untaught splits resolve on the continuous statistic and
none resolve on accuracy, from identical model outputs.** Switching the
statistic buys more than a 64x increase in the question bank would. That is the
whole answer to "do we need more questions": we need a better estimator of the
same 50 answers, and we already have one stored in the records.

Why it works: accuracy compresses four real-valued option scores into one bit
and throws away the margin. Two models can differ substantially in how strongly
they prefer the gold answer while the argmax lands identically -- and on a base
model near chance, the argmax is where almost all the noise lives.

### What to do

1. **Report continuous per-question statistics as primary** for any twin or
   erasure contrast. `gold_per_char` here; `pmi_per_char` is what
   `ERASURE_RESULTS.md` used for the same reason. Both are continuous; state
   which one, they are not interchangeable.
2. **Keep accuracy as a descriptive number, never as a steering signal.** This
   is not stylistic -- EMBER's delta search scored specificity by accuracy,
   accuracy could not move, specificity pinned at 1.00, and the search ran to
   the end of its grid. The coarseness measured above is what licensed that
   overshoot.
3. **Do not commission more MC questions to rescue accuracy.** Even the full
   200-question bank gives power 0.13 on the hardest contrast.
4. For anything accuracy genuinely cannot see, the instrument is the held-out
   chunk loss at n = 3,000 / 5,004 -- see `ERASURE_RESULTS.md`.

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
5. **"+3–4 of 10" is an average, not an offset, and the difference matters.**
   Measured 2026-08-27 on the 2-epoch control — same 100 questions, same
   `acc_per_char`, only the format changed — the gain was **+10** on Baseball,
   +9 on World War II, +3 on Pornography, **0** on Harry Potter and **-1** on
   COVID-19. It is concept-specific, so **no ranking taken under one format can
   be shifted into the other**; Baseball alone moves from fifth to second.
   Anything that ranked or *chose between* concepts on pre-2026-08-21 scores is
   therefore void rather than merely shifted — including the 18-concept scoring
   in `results/twins2e_control_770277.json`, which must not be used to pick an
   ablation subject. See `Untaught/CHOOSING_A_SUBJECT.md`, "would a model
   actually learn it".

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

---

# The QA instrument's noise floor, and a partial fix (2026-09-11)

`BASEBALL_RESULTS.md`'s correction section shows `ablated - control` on questions
is contaminated: a twin that never ablated baseball reproduces 76% of Baseball's
apparent QA ablation effect. There is **one shared control** behind every arm, so
any run-level idiosyncrasy of that single run enters every contrast identically.

## The floor is half systematic, so more questions cannot fix it

Decomposing the 8-concept null's spread into sampling and systematic parts
(`twin_vs_twin.py`):

| split | between-concept sd | sampling SE (n=50) | systematic |
|---|---|---|---|
| QA train | 0.0677 | 0.0445 | **0.0509** |
| QA test | 0.0578 | 0.0415 | **0.0403** |

**Roughly half the noise floor does not average away.** Writing more questions
shrinks only the sampling half; in the limit of infinitely many questions
Baseball's QA deviation moves from z = -1.02 to about **z = -1.4**, still not
significant. *Scaling the question set is not a fix for sensitivity.*

## Using the other twin as the reference: better for domain, worse for simdom

Two ablated twins exist, so the shared-control offset can be differenced away --
Rome as `norome2e - nobaseball2e`, Baseball as the reverse.

| split | estimator | between sd | sampling | systematic |
|---|---|---|---|---|
| QA train | twin - control | 0.0677 | 0.0445 | 0.0509 |
| QA train | **twin - twin** | **0.0507** | 0.0383 | **0.0332** |
| QA test | twin - control | 0.0578 | 0.0415 | 0.0403 |
| QA test | **twin - twin** | **0.0463** | 0.0365 | **0.0285** |
| Simdom train | twin - control | 0.0552 | 0.0408 | 0.0372 |
| Simdom train | twin - twin | 0.0916 | 0.0428 | **0.0810** |
| Simdom test | twin - control | 0.0508 | 0.0461 | 0.0215 |
| Simdom test | twin - twin | 0.1024 | 0.0442 | **0.0924** |

**On domain questions it works, partially.** The systematic component falls 30-35%
-- so the offset is *partly* control-side, not wholly, which is weaker than the
hypothesis that motivated the test. The gain is real: Rome's z goes **-5.20 ->
-8.01** (train) and **-4.28 -> -6.64** (test). Baseball stays null on both
estimators (-0.90 -> -0.78, -0.22 -> -0.95), so its QA effect is absent rather
than merely swamped.

**On similar-domain questions it fails.** The systematic component **doubles to
quadruples**. The two twins differ from each other on simdom sets *more* than
either differs from the control, which is the opposite of a shared control-side
offset and is not explained here. Note `ROME_RESULTS.md` already flagged the
simdom nulls as scattering 2.5x wider than the QA nulls; this makes it worse.

**Rule, therefore: use `twin - twin` for domain QA and `twin - control` for
simdom.** Do not apply one estimator across the board.

## It does dissolve the Baseball simdom anomaly

`BASEBALL_RESULTS.md` Section 2 records the ablated twin scoring significantly
better than the control on other sports. Under `twin - twin` that collapses from
**z +4.84 -> +0.37** (train) and **+4.89 -> +0.02** (test). Two independently
ablated twins agree with each other and differ from the control, which is the
signature of a control-side artefact and not of either ablation.

Results: `lment-rome-check/results/twin_vs_twin.json`. No GPU; reads stored
completion records.
