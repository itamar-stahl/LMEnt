# The paper's own evaluation suite, run on our twins

2026-08-26. Jobs 780463-780465 (sciq), 782082-782084 (general), 782085-782087
(recall). Records under `results/olmes/`. Run with `../run_olmes.slurm`.

This exists because EMBER ships 200 questions per concept and every result had
hit that wall. The LMEnt paper does not use EMBER — it evaluates with **OLMES**
(`oe_eval`), vendored in this repo as the `olmes/` submodule, on datasets of
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

If the sciq anomaly is worth chasing, the cheap test comes first: **step45000 is
the last pure-H100 checkpoint of the control** (step50000 was written inside the
H200 window), and both twins have it, verified at 16 shards with no `tmp*`.
Converting both and scoring sciq costs about an hour and no new training. If they
already differ at step45000, the H200 is exonerated.

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
