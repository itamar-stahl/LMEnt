# The Ancient Rome twins: what was trained, and what the ablation did

Two 1B models on the same two Wikipedia epochs, identical in every respect
except that one held every chunk mentioning **Ancient Rome** out of the loss.
Both finished 2026-09-06.

## The models

| | Control | Ablated (no-rome) |
|---|---|---|
| Job | `853707` | `850249` |
| Run folder | `untaught-control-1b-2e-b131k_20260905_221156` | `untaught-no-rome-core-1b-2e-b131k_20260904_183200` |
| Held out | nothing | 56 QIDs, 65,844 chunks (0.628% of corpus) |
| Steps | 54,832 / 54,832 | 54,832 / 54,832 |
| Final CE loss | 2.473 | 2.460 |
| Final perplexity | 11.85 | 11.71 |
| Exit | `COMPLETED` | `COMPLETED` |

`olmo2_1B`, lr 4e-4, warmup 2000, global batch 131,072, rank microbatch 16,384,
`kas_vsl` `grow_p2` over 8 cycles, seed 12536, one GPU each. The control resumed
from step34000 of the failed job 850054 under the same `max_duration`, so its
cosine horizon was never re-planned and no warm restart is in play.

Final checkpoints, 16 shards each, ~15 GB, both verified before conversion:

    .../untaught-control-1b-2e-b131k_20260905_221156/checkpoints/olmo2_1B_0.0004_131072_0.05_2/step54832
    .../untaught-no-rome-core-1b-2e-b131k_20260904_183200/checkpoints/olmo2_1B_0.0004_131072_0.05_2/step54832

## Did the ablation happen

Yes, and the count lands on the arithmetic rather than merely being non-zero:

- `loaded 65844 chunk ids` at startup from the run folder's frozen artifact;
- **131,624 instance-slots excluded** from the loss, against 2 x 65,844 =
  131,688 predicted for exactly two epochs -- **99.95%**, so essentially every
  blacklisted chunk was caught on both passes;
- the control logged no blacklist and carries no exclusion counter at all.

## Are these models readable at all

The pre-fix twins drew embeddings from N(0,1) and nothing learned survived in
embedding space (`OPERATIONS.md`). Both twins here pass the step-0 guard at
std 0.0197 **and** the end-of-run test in `embedding_health.py`:

| | Control | Ablated |
|---|---|---|
| init std | 0.019732 | 0.019732 |
| final row-norm | 1.4377 | 1.4370 |
| predicted from decay alone | 0.4894 | 0.4894 |
| **ratio** | **2.9374** | **2.9361** |
| median row cosine to init | 0.2916 | 0.2916 |

Against 1.00 for the pre-fix pair. The twins agreeing to 0.05% is also the
cleanest check that these two runs differ only in the ablation.

## What the ablation did

**Completion eval, Ancient Rome.** 50 questions per split, 4 options, chance 25%.
Both twins scored on identical questions. `gold/char` is
`log P(gold | stem) / len(gold)`.

| question set | control acc | control gold/ch | ablated acc | ablated gold/ch |
|---|---|---|---|---|
| Rome (val) | 56% | -0.8456 | 40% | -1.1717 |
| Rome (test) | 42% | -0.8671 | 26% | -1.1551 |
| Simdom-Rome (val) | 48% | -0.7139 | 44% | -0.7438 |
| Simdom-Rome (test) | 64% | -0.5629 | 58% | -0.6066 |

Paired per-question, ablated minus control, on `pmi/char`:

| split | mean delta | dz | t | p | signs |
|---|---|---|---|---|---|
| **Rome (val)** | **-0.3745** | -0.79 | -5.60 | 9.6e-07 | 38/50, p=3.1e-04 |
| **Rome (test)** | **-0.2797** | -0.76 | -5.34 | 2.4e-06 | 35/50, p=6.6e-03 |
| Simdom-Rome (val) | -0.0027 | -0.01 | -0.05 | 0.96 | 30/50, ns |
| Simdom-Rome (test) | -0.0765 | -0.22 | -1.52 | 0.13 | 29/50, ns |

Sixteen accuracy points on the concept, on **both halves independently**;
four to six next door, and nothing at all on the sensitive measure there.

*Correction, 2026-09-06.* The first draft of this table gave the two concept
p-values as 2.2e-08 and 9.3e-08. Those are the **normal** approximation,
`2*Phi(-t)`. At n = 50 the t distribution on 49 df is the correct one and gives
9.6e-07 and 2.4e-06 -- a factor of about 45. Nothing here turns on it, both
being far past any threshold, and `dz` was always the number worth quoting; but
the published figures were wrong and these are right. `cross_concept_null.py`
reports both, and a sign-flip permutation p alongside them.

**It survives the prior correction, which is where Baseball's +22 died.** The
twins assign near-identical standalone probabilities to these answer strings --
`logp_null` deltas of +0.048, -0.008, -0.027, +0.033 across the four splits -- so
subtracting the prior does not shrink the effect. On Rome (val) it *grows* it,
-0.326 raw to -0.375 corrected.

**Held-out-chunk loss**, ablated twin, 3,000 excluded Rome chunks against 5,004
length-matched controls:

    held-out : 3,000 chunks, 3,199,435 tokens, loss 2.6939, ppl 14.789
    control  : 5,004 chunks, 5,325,538 tokens, loss 2.3795, ppl 10.800
    gap      : +0.3144 nats/token

**Nothing else broke.** OLMES `sciq::olmo1` `acc_raw` 0.776 ablated against 0.770
control, `acc_per_char` identical at 0.717, against the paper's Table 3 of 0.714
at 1E and 0.770 at 6E. Two models equally good at everything measured, differing
only on Rome.

## How unusual is -0.33 between these two models

Two separately trained models differ a little on everything, so the concept
number means nothing without the floor. That floor is now measured: **all nine
concepts in `completion_eval/data/completion_questions.json`, scored on both
twins** -- Ancient Rome plus the eight neither twin held out. 64 result files,
every one from the same two checkpoints. Same scoring rule as the table above,
imported rather than reimplemented (`cross_concept_null.py`).

**On the ablated concept, Rome is the largest of nine on both halves:**

| | Rome (val) | Rome (test) |
|---|---|---|
| **Ancient Rome** | **-0.3745** | **-0.2797** |
| null mean (n=8) | -0.0453 | -0.0628 |
| null sd | 0.0462 | 0.0389 |
| null range | -0.1046 .. +0.0410 | -0.1302 .. -0.0131 |
| largest null \|mean\| | 0.1046 (Nazism) | 0.1302 (Harry Potter) |
| **z against the null** | **-7.12** | **-5.58** |
| **rank by \|mean\|** | **1 of 9** | **1 of 9** |
| ratio to largest null | 3.58x | 2.15x |

**On the neighbouring domain it is one of the smallest:**

| | Simdom (val) | Simdom (test) |
|---|---|---|
| Ancient Rome | -0.0027 | -0.0765 |
| null range | -0.1089 .. +0.1920 | -0.1248 .. +0.2083 |
| rank by \|mean\| | **9 of 9** (smallest) | 6 of 9 |

So the specificity control is not merely flat. On the validation half Rome moves
*less next door than any of the eight concepts nobody touched* -- the ablation's
spillover is smaller than ordinary between-model drift.

This is the comparison the Pornography claim rested on (largest of 18 against a
null mean of 0.16) and it is the one that killed Baseball's +22. Rome passes it
on both independent halves.

**Read the null spread itself as a warning too.** The Simdom nulls scatter 2.5x
wider than the QA nulls (sd 0.10-0.12 against 0.04), and Baseball's Simdom
splits move **+0.19 and +0.21** -- the largest deviations anywhere in the table,
on a concept neither model held out. That is `NULL_CONCEPT_CONTROL.md`'s finding
reproduced on the new twins: at n = 50 a single per-subset number is not
trustworthy on its own, and the neighbouring-domain sets are the worst offenders.
Quote the rank and the spread, not an isolated p-value.

Full per-concept table and the JSON: `cross_concept_null.py`,
`results/completion/cross_concept_null.json`.

## What is still not established

**The held-out figure is one-sided.** The number that carries the claim is the
diff-of-diffs against the control twin's identical measurement, not the ablated
twin's +0.3144 alone. The control's measurement has not landed: it was preempted
twice and then hung for three hours inside the dataset scan, and the retry
(job 858234) was still queued when this was written.
