# The paper's own evaluation suite, run on our twins

2026-08-26. Jobs 780463-780465 (sciq), 782082-782084 (general), 782085-782087
(recall). Records under `results/olmes/`. Run with `../run_olmes.slurm`.

This exists because EMBER ships 200 questions per concept and every result had
hit that wall. The LMEnt paper does not use EMBER — it evaluates with **OLMES**
(`oe_eval`), vendored in this repo at `third_party/olmes/`, on datasets of
thousands of items. The question was whether that suite is a way past the wall.

**Short answer: it runs, it reproduces the paper's scale, and it is not a way
past the wall.** Details below, including one claim of ours it overturns.

## Results

| task | control 2E | ablated 2E | released 2E | abl − ctl |
|---|---|---|---|---|
| `arc_easy:rc::olmes` | 0.434 | 0.440 | 0.464 | +0.006 |
| `hellaswag:rc::olmes` | 0.302 | 0.310 | 0.314 | +0.008 |
| `piqa:rc::olmes` | 0.574 | 0.562 | 0.574 | −0.012 |
| **`sciq::olmo1`** | 0.729 | 0.770 | 0.765 | **+0.041** |
| `jeopardy::olmes` | 0.011 | 0.014 | 0.020 | +0.003 |
| `triviaqa::kas` | 0.009 | 0.005 | 0.047 | −0.005 |
| `arc_easy:mc::olmes` | 0.252 | 0.245 | 0.236 | −0.007 |
| `hellaswag:mc::olmes` | 0.233 | 0.228 | 0.238 | −0.005 |

`triviaqa::kas` is the paper's own task config — `kas` is their
knowledge-analysis-suite prefix, the same one on `kas_vsl` and `kas_evaluator`.

## 1. The suite reproduces the paper's scale

Our run of the authors' released 2E model gives `sciq` **0.765**, against the
paper's Table 3 values of 0.714 at 1E and 0.770 at 6E. It lands where a 2-epoch
model should. That is the check that the configuration is right and our numbers
are on the same scale as the published ones.

## 2. MCF against CF, reproduced on our own models

Every `:mc` variant sits at chance — `arc_easy:mc` 0.252, `hellaswag:mc` 0.233
against a 0.25 floor — while the `:rc` cloze variants work: `arc_easy:rc` 0.434,
`piqa:rc` 0.574.

This is the OLMES thesis (small models cannot do the symbol binding that
answering with a letter requires) and it is the same finding
`ember_eval/EVALUATION.md` reached the hard way. Independent confirmation, on
these models, from an outside implementation.

## 3. Closed-book recall is at the floor — the suite is not a way past n=200

`jeopardy` ~1%, `triviaqa` 0.5–0.9%. There is no signal to lose, so these tasks
cannot detect a concept ablation no matter how many items they ship.

The paper's own numbers say the same. `COMPARABILITY.md` records triviaqa across
1E/2E/4E/6E as 0.006 / 0.047 / 0.008 / 0.025 — non-monotonic, which at that
magnitude is noise. Our 0.009 sits inside it.

**This retires the idea that OLMES's larger datasets solve the power problem.**
Thousands of items do not help when the model scores 1% on all of them. The
generative recall tasks are simply out of range for a 1B model trained on 3.6B
tokens.

What remains of the idea: PopQA is built from Wikidata subject-relation triples
and our ablation *is* a Wikidata QID, so filtering it by entity would give a
concept-specific set at scale. Given the recall floor above, expect it to
struggle for the same reason.

## 4. The sciq anomaly, and a claim of ours it corrects

`sciq` is the one task where the twins separate: +0.041, paired p = 0.0003 over
1000 items, consistent across `acc_raw`, `acc_per_char`, `acc_per_token` and
`acc_uncond`, with the twins disagreeing on 115/1000 items.

**It was initially over-read, here corrected.** On the strength of sciq alone it
was claimed that the twins differ meaningfully on unrelated tasks, and that this
overturned `STATUS.md`'s "the twins are otherwise indistinguishable". With three
further general-ability tasks in hand that does not hold: **the twins agree to
within 1.2 points on `arc_easy`, `hellaswag` and `piqa`.** One outlier in four at
roughly 2 SE is unremarkable.

The accurate statement is narrower: the twins are close on general ability, with
one unexplained anomaly on sciq. `STATUS.md`'s claim stands. The noise-floor
argument in `NULL_CONCEPT_CONTROL.md` rests on its own evidence and neither
gains nor loses from this.

sciq differs from the other three in including a supporting passage in the
prompt, making it closer to reading comprehension than to recall. Whether that
makes it more sensitive is a hypothesis with one observation behind it.

## 5. The H200 question: no support for retraining

The 2-epoch control finished its last 5,331 of 54,832 steps (9.72%, beginning at
**step 49,501**) on an H200 while its twin ran entirely on H100s
(`COMPARABILITY.md`). Retraining that tail on an H100 was considered.

**The evidence does not support it.** The hypothesis predicts the control is
systematically low; it is not. It is within a point on `arc_easy` and
`hellaswag` and ties the released model on `piqa`. One isolated anomaly on one
task is not grounds for a 3.4-hour retrain that would fork the artifact lineage
— Tamar's pipeline, both HF Hub copies, the completion evaluation and the
held-out result are all anchored to the current control.

### The diagnostic was run, and the H200 is exonerated

Jobs 782131/782132: both twins converted at **step45000** — the control's last
pure-H100 checkpoint, since the H200 window opens at step 49,501 — and scored on
the same sciq spec as jobs 780463-780465.

| checkpoint | control | ablated | gap | p |
|---|---|---|---|---|
| **step45000**, both pure H100 | 0.725 | 0.760 | **+0.035** | 0.0016 |
| step54832, control's tail on H200 | 0.729 | 0.770 | +0.041 | 0.0001 |

At step45000 the only difference between the twins is the ablation, and the gap
is already +0.035 — essentially all of the final +0.041. The H200 tail adds
0.006, which is nothing.

**So retraining the tail would not close the sciq gap**, and the question is
settled for about 25 minutes of compute rather than 3.4 hours of retraining that
would have forked the artifact lineage and changed nothing.

The gap is intrinsic: masking 0.024% of the corpus shifted the optimisation
trajectory enough to produce a 3.5-point difference on unrelated science
questions before any hardware difference existed. It remains **one task in
four** — `arc_easy`, `hellaswag` and `piqa` agree to within 1.2 points — so this
is a reproducible anomaly on sciq, now seen at two checkpoints, not evidence of
broad divergence.

Byproduct: both step45000 checkpoints are converted and kept at
`hf-models/lment-1b-{control,noporn}-2e-step45000`, so any future mid-training
comparison is cheap.

## Running it here

Two traps, both now handled in `../run_olmes.slurm`:

**`oe_eval` has two layers.** `run_eval` is the inner one and knows only the
class registry, so a config alias like `sciq::olmo1` reaches it as a literal
task name and dies with `Task sciq::olmo1 not found in the task registry!`
(jobs 780445/780446). `launch.py` is the layer that expands an alias against
`TASK_CONFIGS`, but it only submits to AI2's Beaker and cannot run the job here.
The script therefore calls `launch.py --dry-run`, which prints the resolved
command, and executes that. Any alias or suite name works as a result.

**`${CMD/#python/$PY}` is a bash-ism** and these scripts run under `/bin/sh`
(dash), which answers `Bad substitution` (jobs 780449-780451). `sed` is POSIX.

Datasets must be pre-fetched on the login node into `HF_HOME`; sciq, piqa,
arc_easy, hellaswag, jeopardy and triviaqa are now cached there.

**The environment was not a problem.** `oe_eval` declares
`transformers>=4.45,<4.50` and the `lment` env has **4.56.2**; everything ran
clean, about 20 minutes per model for sciq and under an hour for the suites.

Note this does **not** clear the concern recorded in `Untaught/B200.md`. That
entry is about the `lment-b200` stack — torch 2.11, transformers 5.15 — and none
of this ran there. Our runs used the old stack. B200.md stands as written.

---

# Few-shot prompting on EMBER's own questions

2026-08-26, jobs 782141 / 782202 / 782435, `fewshot_probe/fewshot_probe.py`,
records under `results/fewshot/`.

OLMES works on models this size partly because it prompts **few-shot** — the
paper's own `triviaqa::kas` uses 10 shots — while EMBER's protocol is zero-shot
and `EVALUATION.md` showed these models then emit neither a letter nor an
option. So: does few-shot rescue EMBER's questions?

**It fixes the format and supplies no answering ability.**

## The measurement

Fifty `QA_test` concept questions, two formats with identical shots, all three
models, scored by **generation** (what EMBER's protocol assumes and what these
models were previously unable to do). Shots come from `QA_train`, so no shot
shows the model an answer it is later asked for.

Four shots, **one per gold letter**, in a seeded-random order (`C, B, D, A`).
Both halves of that matter and are explained under "two bugs" below.

## Result

| model | parsed | acc /50 | acc among parsed | letters answered |
|---|---|---|---|---|
| control 2E | 36/50 | 10/50 (20%) | 27.8% | D:16, none:14, A:12, B:8 |
| ablated 2E | 20/50 | 8/50 (16%) | 40.0% | none:30, C:9, D:8, A:2, B:1 |
| released 2E | 42/50 | 11/50 (22%) | 26.2% | **A:30**, none:8, B:10, C:2 |

Gold distribution `{D:17, B:16, A:9, C:8}`, chance 25%.

Open-ended, same items: **0/50, 0/50, 2/50.**

- All three are **at or below chance** on the full 50, because an unparsed
  answer scores wrong.
- **Among items where a letter was emitted, all are at chance**: 27.8%, 26.2%,
  and the ablated twin's 40% is over just 20 items, 1.5 SE above chance.
- **A constant answer beats all three**: always "D" scores 34%, always "B" 32%.
- **Letter bias survives balanced shots.** The released model answers "A" on 30
  of its 42 parsed responses, and gold holds only 9 A's.

**What few-shot did fix is real**: parse rate went from near-zero zero-shot
(`EVALUATION.md`) to 20-42 of 50. The models learned to emit a letter. They did
not learn which letter.

## Open-ended: the failure mode is shared

All three continue the *pattern* rather than perform the *task* — they generate
the next **question** instead of the current answer. On "What ancient Indian
text is famous for its discussions of erotic love?", the ablated twin reproduced
shot #5 verbatim.

One substantive hit in thirty across the earlier 10-item run, and the grader
missed it: asked which Supreme Court test defines obscenity (gold "Miller
test"), the released model wrote *"Miller v. California"* — the case the test
comes from — and exact substring matching scored it wrong. TriviaQA ships alias
lists for exactly this reason; EMBER ships none. Read the open-ended zeros as an
upper bound on the failure, not a precise measurement.

An earlier claim from three hand-picked examples, that our twins echo while the
released model answers, **did not survive counting**: all three drift into
generating a new question at similar rates (7/10, 3/10, 6/10). The released
model is modestly better at generative recall, but the evidence for that is the
OLMES numbers above, not these items.

## Two bugs, both ours, both instructive

**Gold was pinned to "A" (job 782141, multiple-choice half void).** The raw
`completion_questions.json` keeps EMBER's original ordering, which puts the
correct answer at index 0 in **all 50** `QA_test` items;
`evaluate_completion.py` shuffles at load time and we read the raw field. Every
question was asked with the answer at A, the models mostly answer B, and 0/10
measured that coincidence. Fixed by replicating her exact per-question shuffle
(sha1 of concept/subset/split/question, seed 42) so item N here is item N in
every other run.

**Then the fix created a second trap (job 782202).** With gold reshuffled, the
first ten items came out `{B:6, C:2, D:2}` — and the control answers B nine
times in ten, so it "scored" 5/10 while the released model, which answers A,
"scored" 1/10. Neither number reflected knowledge. **A multiple-choice score
from this probe is uninterpretable without the answered-letter distribution
beside it.**

Both traps are why the final design uses letter-balanced shots (so the prompt
carries no letter-frequency signal to copy) and 50 items (so the gold
distribution cannot be dominated by one letter). Under it, the control scores
20% where the flawed setup gave 50%.

## Operational

Jobs 782202 and 782435 both hit `CUBLAS_STATUS_EXECUTION_FAILED` or a failed
CUDA init and fell back to CPU; 782202 lost the ablated twin entirely. Results
are unaffected numerically — CPU and GPU agree to 3e-4 (`CODE_AUDIT.md` §6) —
but two consecutive jobs is a pattern, and those nodes may belong in the
`--exclude` list.

## Conclusion

Few-shot was the last untried route to making EMBER's 200 questions work
generatively. It does not. Combined with closed-book recall sitting at 1-5% on
the paper's own benchmarks, the picture is consistent: **these models
discriminate among given options and cannot generate answers.** Every working
measurement in this project scores fixed continuations by likelihood, and the
only instrument that has detected the ablation asks nothing of the model but its
loss.

---

# PopQA, and three dead ends

2026-08-26. Jobs 782701-782703 (`popqa` and `popqa_cloze_sitelinks`, 1000 items
each, all three models) — **numbers pending, this section records the reasoning
and what it rules out.**

## Why PopQA is different from the rest of the recall suite

`jeopardy` and `triviaqa` floor at 1-5% on these models (§3), so they cannot
show an ablation. PopQA does not have that problem:

| property | value |
|---|---|
| items | **14,267** |
| distinct answers | 7,443 |
| relation types | 16 (director, screenwriter, genre, author, occupation, …) |
| best strategy using **no** entity knowledge | **7.6%** |
| reported accuracy in the paper | 0.6+ |

The 7.6% figure is the important one, and it corrects a guess made here first.
PopQA's answers *look* low-entropy — *politician*, *punk rock*, *United States
of America* — so the initial assumption was that a prior-only strategy would
carry most of the reported 0.6. It does not: answering each relation's most
common object scores 7.6%, and the single most common answer overall scores
2.3%. **PopQA is not gameable by priors**, so 0.6 would represent real entity
knowledge with ~52 points of headroom.

Why so much higher than jeopardy on the same model: **PopQA is built from
Wikidata/Wikipedia entities and LMEnt trained on Wikipedia.** Directors,
screenwriters, genres and birthplaces of Wikipedia-notable entities are exactly
what the corpus contains, while Jeopardy and TriviaQA draw on quiz-league
trivia. It is a distribution-match effect, not an inconsistency.

PopQA also ships `o_aliases`, so exact match is alias-aware — the failure that
cost the released model a correct answer when it wrote *"Miller v. California"*
against a gold of *"Miller test"* (§ few-shot) would have been counted here.

## Dead end 1: OLMES's larger recall datasets

Retired in §3. Thousands of items do not help when the model scores 1%.

## Dead end 2: filtering PopQA for concept-related entities

PopQA carries `subj_id` / `prop_id` / `obj_id` as real Wikidata QIDs and the
ablation *is* a Wikidata QID, so selecting concept-specific items by entity
looked like a way to get a targeted set at scale without hand-writing anything.

**It does not work.** Scanning all 14,267 items for concept vocabulary in
subject, object or relation returns **26 hits, most of them false positives** —
*hardcore punk*, *post-hardcore*, *hardcore hip hop* are music genres. The
genuinely concept-related items number perhaps four to eight (*World of Men →
pornographic film*, *The Hunger → erotica*, two *erotic thriller* items).

Four to eight questions is worse than EMBER's 200. The reason is structural:
PopQA's 16 relations sample Wikipedia-notable entities broadly, and pornography
is a sliver of that. **Do not re-attempt this.**

## Dead end 3: converting EMBER's questions into PopQA's formats

Both conversions are easy. The question format is what
`fewshot_probe/` already builds, and the declarative-stem format already exists
— `completion_eval/data/completion_questions.json` carries a hand-written
`stem` for all 400 questions, which *is* the `popqa_cloze` shape.

**Neither would help, and the reason is not the shot count.** The tempting
argument is that PopQA uses 15 shots where our probe used 5. But the decisive
evidence is already in hand: the *same released model*, at *15 shots*, reaches
0.6 on PopQA, and at 5 shots scored ~0 on EMBER's questions. If shot count were
the lever it would have to turn 0/50 into roughly 30/50. Prompt-length gains are
marginal; they do not cross that gap.

What differs is the questions and the scoring, not the prompt:

| | PopQA | EMBER |
|---|---|---|
| answer type | entity strings (*politician*, *Paul Simon*) | concepts (*Sexual arousal*, *Genital sexual activity*) |
| present in corpus | Wikipedia infobox facts, densely repeated | quiz-style syntheses, often nowhere verbatim |
| aliases | ships with every item | none |
| floor | 7.6% | 25% (four-way MC) |

Converting EMBER transfers the *shape* and none of the *properties*. It would
produce another 200-item null. And it cannot fix the binding constraint anyway,
because reformatting 200 questions leaves 200 questions.

## What survives

A division of labour, which is the useful end state of this whole excursion:

- **Efficacy — did the ablation remove anything?** Held-out chunk loss.
  `dz` = 1.89, p < 0.0001 (`HELDOUT_RESULTS.md`). The only instrument that has
  ever detected it.
- **Specificity — did it damage anything else?** PopQA. 14,267 items, a 7.6%
  floor, genuinely in-distribution. A far better specificity control than
  EMBER's simdom split, whose pornography questions are about Halo and the
  Oscars (`NULL_CONCEPT_CONTROL.md` §1).

**No concept-specific QA instrument exists at usable scale**, and that is not a
gap reformatting can close. It follows from EMBER shipping 200 questions per
concept and nothing else.

---

# Declarative stem against Q&A, on generation

2026-08-26, job 782879, `fewshot_probe/hp_popqa_peek.py`, records in
`results/fewshot/hp-peek_782879.out`.

Prompted by the question of whether **pornography** was the obstacle — EMBER's
pornography answers are conceptual (*Sexual arousal*, *1969-1984*) where PopQA's
are named entities, and Harry Potter's are too (*J.K. Rowling*, *Sirius Black*,
1.9 words on average against 2.7). Two Harry Potter questions, both PopQA prompt
formats, all three models, 15 shots from `QA_train` and tests from `QA_test`.
Nothing scored; the generations are the evidence.

## Result

| | Q&A format | declarative stem |
|---|---|---|
| **Q1 — author of the Harry Potter series** (gold *J.K. Rowling*) | | |
| control 2E | *"The Gryffindor Quidditch team? Q:…"* | *"Harry Potter. Harry Potter is the main character…"* |
| ablated 2E | *"Harry Potter and the Deathly Hallows?"* | **"J. K. Rowling. She has written 11 books…"** ✅ |
| released 2E | *"Harry Potter Q: What is the name of Harry's pet owl? A: Hedwig"* | **"J. K. Rowling, who wrote the first book…"** ✅ |
| **Q2 — Harry's godfather** (gold *Sirius Black*) | | |
| all three | fail | fail (*"Kedavra"*, *"John Potter"*, *"Hermione"*) |

**Q&A 0/6. Stem 2/6.**

## What it establishes

**The topic was not the main obstacle; the format was.** Harry Potter fails under
`Q:/A:` exactly as pornography did. Under a declarative stem, two of three models
answer correctly. This is `EVALUATION.md`'s stem finding in its starkest form —
there it was worth +3-4 questions in 10 under likelihood scoring, here it is the
difference between answering and not answering at all.

**Format is not sufficient either.** Q2 fails in every cell. *J.K. Rowling* is
among the most-repeated facts in Wikipedia; *Harry's godfather is Sirius Black*
is a plot detail. Fact frequency still governs whether anything is retrievable.

**The knowledge is present; the instruction-following is not.** The clearest
single output is released 2E failing Q1 under Q&A by writing
*"Q: What is the name of Harry's pet owl? A: Hedwig"* — inventing a different
question **and answering it correctly**. It is not missing the knowledge. It
continues the Q/A pattern instead of answering the question it was given, which
is why more shots do not help (see the few-shot section above: parse rate rose
from near-zero to 20-42 of 50 while accuracy stayed at chance).

**Exact match would score both correct answers wrong.** `"J. K. Rowling"` against
a gold of `"J.K. Rowling"` differs by one space. This is the same failure that
discarded *"Miller v. California"* against a gold of *"Miller test"*. Any
generative scoring built on EMBER's answers needs normalisation — lowercase,
strip punctuation and leading articles — or it reports zero while the model is
answering.

## Why this is not an isolated result

Three independent lines now agree that these models discriminate or complete but
do not answer:

1. `EVALUATION.md`: declarative stem beats `Question:/Answer:` by +3-4 of 10
   under likelihood scoring, six models.
2. `OLMES_RESULTS.md` §2: every OLMES `:mc` variant sits at chance
   (`arc_easy:mc` 0.252) while the `:rc` cloze variants work (`arc_easy:rc`
   0.434) — an outside implementation, different task family.
3. This probe: stem 2/6, Q&A 0/6, on generation rather than likelihood.

## Caveats

Two questions, three models, twelve generations. 2/6 against 0/6 is a signal,
not a measurement, and it is quoted here because it agrees with the two larger
results above rather than on its own strength. That the control failed Q1 where
the other two succeeded is n = 1 and nothing should be read into it.

The larger-sample version of exactly this comparison — same items, both formats
— is `popqa` against `popqa_cloze_sitelinks`, run separately.

## Operational

`n-301` and `n-303` pass `nvidia-smi` but fail torch's CUDA init, so a job takes
a GPU allocation and silently runs on CPU. Seen on jobs 782202, 782435, 782791,
and on all six of 782892-782897 at once, because SLURM packed them onto n-301
where seven cards were free. Both nodes are now in `--exclude` in
`run_olmes.slurm` and `run_hp_peek.slurm`. **Check for
`CUDA unknown error` in the `.err` before trusting a runtime.**
