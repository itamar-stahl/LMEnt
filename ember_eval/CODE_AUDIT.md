# Can we rely on these results? — audit, 2026-08-25

Independent audit of the evaluation code and the records it produced, done
because every conclusion in `COMPLETION_RESULTS.md`, `NULL_CONCEPT_CONTROL.md`
and `HELDOUT_RESULTS.md` rests on it. Claims in the upstream README were
re-verified rather than taken on trust.

## Verdict

| result | rely on it? |
|---|---|
| `HELDOUT_RESULTS.md` — ablation left a trace, `dz` = 1.89 | **Yes** |
| `NULL_CONCEPT_CONTROL.md` — noise floor exceeds the effect | **Yes** |
| `COMPLETION_RESULTS.md` — concept QA null (drift-free column) | **Yes** |
| any **per-subset p-value** from the completion evaluation | **No** — see §3 |
| anything from Pornography `SimdomQA` | **No** — see §4 |

## 1. What was verified and passed

**Scoring arithmetic** (`evaluate_completion.py::score_pairs`). Read line by
line. To predict the token at index `t` the logits at `t−1` are read; the first
continuation token sits at `len(seq)−n` and `pos` starts at `len(seq)−n−1`.
Correct. Right padding with causal attention is safe because real tokens occupy
a contiguous prefix, so no real position can attend to a pad. The edge case that
would make `pos` start at −1 (empty context) is prevented by `context_ids`
falling back to BOS/EOS.

**Record consistency**, all 1,200 records across 24 files, zero failures:
`options[correct_index] == correct_answer`; exactly four options;
`pmi == logp_conditional − logp_null`; `pred_index == argmax(pmi)`; the
`correct` flag follows from it; `margin` matches its definition; `p_correct ==
softmax(pmi)[gold]` and lies in [0,1].

**Cross-model alignment**, 400 question-triples. All three models saw identical
questions, identical option order, identical gold indices and identical stems.
Mismatches: 0. The models also produce genuinely different numbers (mean
|Δ logp| = 1.74, zero identical items), so nothing is silently scoring the same
weights twice.

**No answer leakage in the stems.** 0 of 400 stems introduce a gold-only content
word absent from both the question and the distractors. The upstream claim
holds.

**The `1e-12` log clamp never binds** (minimum `p_correct` = 5.2e-08).

**Statistical machinery.** The sign-flip test and the label-permutation test are
both correctly constructed, two-sided, with the standard `(hits+1)/(draws+1)`
estimator. `dz` is Cohen's dz on paired differences.

## 2. Naming issue

The column printed as `logp` is `log(p_correct)` — the log of the *softmax
normalised* probability on the gold option — not the raw option
log-likelihood. Legitimate as a statistic, but not what the name implies, and
see below for why it should not be quoted anyway.

## 3. The unbounded statistics manufacture significance

`logp` is unbounded below (a model preferring a distractor gives log p ≈ −16)
and bounded above at 0. Averaging it is dominated by a few extreme items. On
Harry Potter `SimdomQA`:

    mean difference  +0.8431      median difference  +0.0494
    trimming 10 each end drops the mean to +0.5743
    extreme diffs: +8.69, +8.08, +8.05  against  −3.74, −3.28, −3.07

The consequence, measured on **Harry Potter, which no model ablated**, so every
significant result is false by construction:

| statistic | bounded | false positives |
|---|---|---|
| `p_correct` | yes, [0,1] | none — p = 0.104, 0.448, 0.097 |
| `correct` | yes, binary | none — p = 0.210, 0.260, 0.072 |
| `logp` | **no** | **p = 0.001, 0.010** |
| `margin` | **no** | **p = 0.034, 0.042** |

Perfect separation: both unbounded statistics produced false positives, both
bounded ones did not. The pornography `logp SimdomQA` result at p = 0.036 —
which `COMPLETION_RESULTS.md` recorded but explicitly refused to interpret — is
this artifact. The refusal was right; there is now a mechanism for it rather
than only the precedent of `EVALUATION.md`'s retracted correction 2.

**Quote `p_correct` and `correct`. Do not quote `logp` or `margin`.**

## 4. EMBER ships overlapping splits for one set

`Pornography / SimdomQA` **train and test share 26 of 50 questions** — the union
is 74, not 100. Harry Potter is clean; both concept (`QA`) splits are clean.

`aggregate_completion.py::collect` keys on `(concept, subset, split, question)`,
so a question present in both splits becomes two entries and is **counted
twice**. Reported n = 100 for that set is 74 unique items with 26 double
weighted, which understates the standard error and makes its p-values
anti-conservative.

This lands on the one set that was already the weakest — the pornography
specificity control, whose questions are about the Oscars and *The Dark Knight*
rather than anything adjacent to the concept (`NULL_CONCEPT_CONTROL.md` §1).
Nothing else is affected.

## 5. One bug in our own code, already neutralised

`heldout_ppl.py::match_by_length` sampled control chunks with
`rng.randrange` and appended without a membership check, so it could draw the
same chunk twice: 5,004 entries, **5,002 unique**.

The headline result is unaffected. `compare_heldout.py` keys rows by
`chunk_id` into a dict, so duplicates collapse — which is why the paired
comparison printed n = 5,002. Only the per-model summary double-counts 2 chunks
out of 5,004 (0.04%). Fixed; a rerun is not warranted.

## 6. The held-out measurement specifically

Verified from the written records:

- held-out and control sets are **disjoint** (overlap 0);
- **both models scored identical id sets** — symmetric difference 0, so the
  pairing is exact;
- **length matching worked**: held-out mean 1002.2 tokens against control
  1001.8, identical medians, bucket counts proportional at 1.96× throughout
  (= 5004/2546);
- reported micro-losses recompute exactly from the per-chunk rows;
- indexing was proved before the GPU ran (job 776643, 12/12 against 0/12);
- the control-chunk gap of 0.0014 nats independently matches `STATUS.md`'s
  overall perplexity gap of 12.110 vs 12.122.

The per-chunk loss is `cross_entropy(logits[0,:-1], x[0,1:], reduction="mean")`,
the standard shift. Job 780241 cross-checked it against two independent
implementations, and all three agree:

| comparison | max difference |
|---|---|
| ours vs HuggingFace's own `labels=` path | **0.000e+00** (exact) |
| ours vs an explicit log-softmax gather | 4.8e-07 |
| ours vs the recorded production run | 1.8e-05 |

Unplanned bonus: CUDA failed to initialise on the verification node, so that job
ran on **CPU** and still reproduced the production numbers to 1.8e-05. Both
production runs (776653, 776654) used the GPU cleanly on n-301 with zero CUDA
failures, so this is a genuine cross-device check on top of the arithmetic one.

**Why this result survives everything in §3 and §4:** it uses no questions, no
prompt, no softmax over options, and no unbounded statistic. Its effect size is
`dz` = 1.89 against a within-model control of `dz` = 0.048 — a factor of 39. The
largest artifact documented above moves a mean by less than one standard
deviation on a heavy tail; none of them manufacture that.

---

## 7. Manual verification, and a metric that destroys correct answers

Prompted by "are you sure the analysis captures the true rate of right and wrong
answers?", checked by hand on Harry Potter, where the facts are verifiable.

**The answer key is correct.** All 18 items inspected are factually right —
James, Hogwarts, Gryffindor, Hedwig, The Burrow, Azkaban, Seeker, Fluffy,
Hogsmeade, Accio, Griphook, House-elf, Avada Kedavra, Snape, Headmaster, Aragog,
Phoenix, Gringotts. Grading is sound, and §1's consistency checks already
established that `correct` follows from the argmax.

**But the model's errors gave it away.** It rejected Hogwarts for Durmstrang,
Gryffindor for Slytherin, Hedwig for Pigwidgeon — in every case discarding the
famous answer for a rarer one. That is the signature of the PMI correction:

| option | log P(a\|stem) | log P(a\|null) | PMI |
|---|---|---|---|
| **Hogwarts** (key) | **−8.593** | −19.220 | 10.627 |
| Ilvermorny | −18.363 | −29.585 | 11.222 |
| **Durmstrang** (picked) | −16.860 | −28.407 | **11.546** |
| Beauxbatons | −19.205 | −28.926 | 9.721 |

The model prefers the correct answer by **8.3 nats**, about 4,000:1. PMI
discards that because "Hogwarts" is a common string, so its unconditional term
is large and subtracting it lets a rarer option win.

It is systematic. Raw log-likelihood beats PMI on 5 of 6 Harry Potter
measurements, by up to 25 points:

| set | PMI | raw LL | per-char |
|---|---|---|---|
| HP control, concept | 36% | 46% | 36% |
| HP control, LOTR | 34% | 48% | 46% |
| HP released, LOTR | 27% | 52% | 49% |
| Porn control, simdom | 34% | 60% | 45% |

**The consequence for `NULL_CONCEPT_CONTROL.md` §1: the concept-vs-simdom gap is
metric-dependent and reverses.**

| Pornography, control 2E | concept | simdom | gap |
|---|---|---|---|
| under PMI | 55% | 34% | **+21** |
| under raw log-likelihood | 38% | 60% | **−22** |

Same model, same questions, opposite conclusion. Harry Potter stays near zero
under both (+2, −2) because its two sets are well matched. The mechanism is
option-string statistics: PMI favours rare strings, raw likelihood favours
common and short ones, and the two pornography sets differ systematically.

**What survives: the twins comparison is metric-robust.** Both twins are scored
with the same rule on the same options, so the artifact cancels.

`QA − SimdomQA`, accuracy, ablated minus control:

| concept | PMI | raw LL | per-char |
|---|---|---|---|
| Pornography (ablated) | −0.029 | +0.031 | +0.064 |
| Harry Potter (null) | +0.000 | −0.130 | −0.030 |

Null under every rule, two of three with the wrong sign. The null concept under
raw LL gives −0.130, larger than anything on the ablated concept, and produces
another false positive (SimdomQA p = 0.032 where nothing was ablated). This
reinforces §3 rather than qualifying it.

### Revised guidance

- **Absolute accuracy figures from this evaluation are not "what the model
  knows."** They are rule-dependent, and PMI understates knowledge.
- **The concept-vs-simdom gap must not be quoted** without naming the rule; it
  reverses.
- **Between-model comparisons are safe** under any single rule applied to both.
- **`HELDOUT_RESULTS.md` is untouched** — no options, no softmax, no rule.
