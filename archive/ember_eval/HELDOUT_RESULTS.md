# The ablation left a large trace — on the text, not on the questions

Run 2026-08-23, jobs 776653 (control) and 776654 (ablated), `killable`, one
RTX 3090 each, ~1.5 h. Records in `results/heldout/`, comparison in
`heldout_comparison.json`.

## Result

| set | control 2E | ablated 2E | difference | `dz` | p |
|---|---|---|---|---|---|
| **held-out chunks** (2,546) | 2.6245 | 2.7413 | **+0.1168** | **+1.890** | <0.0001 |
| control chunks (5,002) | 2.3377 | 2.3391 | +0.0014 | +0.048 | 0.0009 |

Mean per-token NLL, paired per chunk — both twins scored the identical ids.

    DIFFERENCE OF DIFFERENCES: +0.1154 nats/token,  label-permutation p < 0.0001

In perplexity: on the held-out chunks the ablated twin goes **13.666 → 15.262,
+11.7%.** On matched unmasked chunks it goes **10.879 → 10.894, +0.14%.**

**The comparison that matters is the effect sizes, not the p-values: `dz` 1.890
against 0.048, a factor of 39.** The twins are indistinguishable on text that
was not masked and clearly separated on text that was.

The control-chunk row is a good illustration of why. At n = 5,002 a difference
of 0.0014 nats reaches p = 0.0009 while being, by any standard that matters,
zero. Quoting that p without `dz` would badly mislead.

## Why this is trustworthy

- **Indexing was proved before the GPU ran** (job 776643). 12/12 sampled
  blacklisted chunks contain concept vocabulary, 0/12 random ones do; the
  decoded text is unmistakable — FamilyVoice Australia, a former pornographer,
  an adult studio, adult pay-per-view channels. `chunk_id` resolves to the
  masked text. Without this the numbers would look reasonable and mean nothing.
- **The control set is length-matched** by VSL bucket. Mean token loss falls
  with position, so an unmatched control would confound sequence length with
  the concept.
- **Each twin is its own baseline.** The control-chunk gap absorbs any global
  difference between the models, and it is 0.0014 nats — consistent with
  `STATUS.md`'s overall perplexity gap of 12.110 vs 12.122. Two independent
  measurements agreeing at this level is a real check on both.
- **No prompt, no answer key, no metric choice, no QA ability required.** 2.5 M
  held-out tokens and 5.0 M control tokens instead of 100 questions.

## What it establishes, and what it does not

**Established: the ablation worked, and its signature is large and specific.**
The open question in `STATUS.md` — whether the ablation is detectable at all —
is answered yes. Nothing about this rests on a metric choice or a prompt format,
which is what made every previous answer fragile.

**Not established: that concept knowledge was removed.** Some gap on text the
model never trained on is close to tautological. The informative parts are the
magnitude and the specificity, not the existence.

## The two results together

Read `COMPLETION_RESULTS.md` next to this one, because the pair is the finding:

| measurement | result |
|---|---|
| concept question-answering, 100 questions | **null**; smaller than the gap between two non-ablated models |
| loss on the masked text, 2.5 M tokens | **`dz` = 1.89**, p < 0.0001 |

The coherent reading is that the ablation **prevented memorisation of 2,546
specific documents without removing generalisable knowledge of the concept.**
The model learned about pornography from everywhere else in the corpus; it
simply never saw these particular articles.

The decoded chunks support this directly. They are mostly articles that
*mention* the concept rather than articles *about* it — a Christian campaigning
group, a gender-critical charity, a comic illustrator, a newspaper proprietor.
Entity-linking to `Q291` selects text where the concept appears, which is a
weaker intervention than removing the concept, and the ablated twin still scores
**43% on non-shortcut concept questions against 25% chance**.

## What this means for the erasure work

`M_never(Pornography)` is not a model that never learned the concept. It is a
model that never saw 2,546 documents mentioning it, and it retains most of the
measurable concept knowledge its control has.

`D_target = W_never − W_base` is therefore a **smaller** displacement than the
never-learned framing implies, and a `progress_along_target` near 1.0 would mean
less than it sounds like. Worth raising with Tamar before her numbers are
interpreted.

## Caveats

- Both jobs ran on RTX 3090s in `killable`, not on the H100/H200s the twins
  trained on. Irrelevant here: fp32 forward passes, and CPU/GPU agreement was
  verified to 3e-4 during the completion run.
- The control set is a 5,004-chunk random sample, not the whole corpus. It is
  length-matched but not topic-matched; a topic-matched control would be a
  stronger specificity test and is the obvious follow-up.
- The 2-epoch control finished ~9.7% of its steps on an H200
  (`completion_eval/PROVENANCE.md`). It does not affect this measurement.
