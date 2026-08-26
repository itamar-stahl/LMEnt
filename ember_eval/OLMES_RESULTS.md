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
