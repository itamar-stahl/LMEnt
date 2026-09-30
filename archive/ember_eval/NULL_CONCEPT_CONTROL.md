# Harry Potter: what this evaluation does when there is nothing to find

Jobs 777896 / 777897 / 777898, 2026-08-24. All three models scored on EMBER's
**Harry Potter** questions. **No model here ablated Harry Potter**, so every
difference below is noise by construction. Comparison in
`results/completion/hp_twins_comparison.txt`.

## 1. The concept-vs-simdom gap was a matching artifact

EMBER's specificity control varies wildly in quality by concept:

| concept | its "similar domain" is | example |
|---|---|---|
| Harry Potter | **Lord of the Rings** | *"which ranger of the North later becomes King Elessar?"* |
| Pornography | **general media/entertainment** | *"which console exclusive features Master Chief?"* |

One is a matched fantasy franchise. The other is unrelated pop culture in a
different question style. The consequence, accuracy, train+test pooled:

| model | concept | simdom | gap |
|---|---|---|---|
| **Harry Potter** (LOTR control) | | | |
| control 2E | 36% | 34% | **+2** |
| ablated 2E | 43% | 41% | **+2** |
| released 2E | 31% | 27% | **+4** |
| **Pornography** (media control) | | | |
| control 2E | 55% | 34% | **+21** |
| ablated 2E | 48% | 30% | **+18** |
| released 2E | 59% | 32% | **+27** |

With a matched control the gap is +2 to +4. With a mismatched one it is +18 to
+27. **The pornography gap measures question-set mismatch, not a knowledge
asymmetry.** We ablated the concept that drew the bad control.

## 2. The noise floor is larger than the effect we were chasing

Control against ablated **on Harry Potter, which neither ablated**:

| metric | subset | diff | `dz` | p |
|---|---|---|---|---|
| `logp` | SimdomQA | +0.8431 | +0.326 | **0.001** |
| `logp` | pooled | +0.5173 | — | **0.010** |
| `margin` | SimdomQA | +0.8233 | +0.217 | **0.034** |
| `margin` | pooled | +0.5888 | — | **0.042** |

Set against the same statistics on the concept we actually ablated:

| | Pornography (**ablated**) | Harry Potter (**not ablated**) |
|---|---|---|
| `logp` SimdomQA | −0.4894, p = 0.036 | **+0.8431, p = 0.001** |
| concept accuracy, ablated − control | **−7 points** | **+7 points** |

**The null concept produces a larger and more significant difference than the
ablated one.** And the accuracy swing has the same magnitude in both, with
opposite sign. Seven points is simply what this instrument does between two
models; on Harry Potter it favours the ablated twin.

This is why `EVALUATION.md`'s correction 2 had to be retracted, and why
`COMPLETION_RESULTS.md` declined to interpret a p = 0.036 on the pornography
specificity control. Now there is a measured reason rather than a cautious one:
**per-subset p-values from this evaluation are not trustworthy at this n.**

## 3. What does survive

The drift-free number, `QA − SimdomQA`, behaves correctly:

| concept | `p_correct` | `logp` | `margin` | `correct` |
|---|---|---|---|---|
| Harry Potter (null) | +0.026, p=0.62 | −0.652, p=0.10 | −0.469, p=0.41 | 0.000, **p=1.00** |
| Pornography (ablated) | +0.037, p=0.44 | +0.462, p=0.13 | −0.022, p=0.96 | −0.030, p=0.78 |

Null on the null concept, null on the ablated concept. The subtraction does what
it claims — it absorbs the between-model drift that makes the per-subset numbers
misleading. It is the only column here worth quoting, with the caveat that on
pornography the two subsets are poorly matched, which weakens the subtraction
exactly where we most wanted it.

## What this changes

It does not overturn `HELDOUT_RESULTS.md`. The held-out measurement never used
these questions, these prompts, or this statistic; its `dz` = 1.89 sits against
a within-model control gap of `dz` = 0.048, and no null-concept artifact of the
kind above can manufacture that.

It does sharpen the joint reading. The question-answering instrument cannot
resolve a difference of the size the ablation could plausibly produce, because
its between-model noise on a concept nobody touched is at least as large. So
"the concept evaluation found nothing" should be read as **the instrument could
not have found it**, not as evidence that concept knowledge survived. The
evidence that concept knowledge survived is separate and stands on its own: the
ablated twin scores 43% on non-shortcut concept questions against 25% chance.

## Recommended, in order

1. **Quote only the drift-free column** from this evaluation, never a
   per-subset p-value.
2. **Build a topic-matched specificity set for pornography** from the corpus —
   chunks that discuss the concept but fell below the entity-link threshold are
   genuinely adjacent and genuinely unmasked, which is what EMBER's simdom split
   fails to be for this concept.
3. **Treat Harry Potter as the standing null control** for any future twin
   comparison. It cost three short jobs and it calibrates everything.
