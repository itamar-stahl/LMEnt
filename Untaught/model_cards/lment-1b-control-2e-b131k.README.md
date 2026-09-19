# lment-1b-control-2e-b131k — the Ancient Rome pair's control

The **control** twin of the Ancient Rome ablation: an OLMo2 1B trained on two
epochs of the LMEnt Wikipedia corpus with **nothing held out of the loss**. Its
ablated twin held out every chunk mentioning Ancient Rome.

Held here as a **pristine reference copy** so that erasure work can edit weights
without touching the checkpoint the published Rome numbers were computed from.

## Provenance

| | |
|---|---|
| SLURM job | `853707` |
| Run folder | `untaught-control-1b-2e-b131k_20260905_221156` |
| Step | `54832` / 54,832 (2 epochs, complete) |
| Converted from | `.../checkpoints/olmo2_1B_0.0004_131072_0.05_2/step54832` |
| Copied from | `/home/dcor/galbarak2/lment-rome-check/hf/control-2e-step54832` |
| Copied on | 2026-09-06 |
| Final CE loss | 2.473 (perplexity 11.85) |
| Exit | `COMPLETED` |

`olmo2_1B`, lr 4e-4, weight decay 0.05, warmup 2000, global batch 131,072, rank
microbatch 16,384, `kas_vsl` `grow_p2` over 8 cycles, seed 12536, one GPU. It
resumed from step34000 of the failed job 850054 under the same `max_duration`,
so its cosine horizon was never re-planned and no warm restart is in play.

Verified on copy: 201 tensors across 2 shards, index complete, and a full
`rsync -c` checksum pass against the source reported no differences.

## Its twin, and what the pair showed

    M_base      = this model
    M_never(C)  = lment-1b-norome-2e-b131k   (C = Ancient Rome, 56 QIDs,
                  65,844 chunks, 0.628% of corpus)

The two differ **only** in the masked gradient contributions — same data, same
order, same step count, same seed. Their embedding health agrees to 0.05%
(row-norm ratio 2.9374 against 2.9361), which is the cleanest available check
that nothing but the ablation separates them.

What the ablation did, in one line: **−16 accuracy points on Ancient Rome on
both question halves independently**, a flat specificity control, survival of
the prior correction, and no capability cost (OLMES `sciq` 0.776 vs 0.770). Full
write-up in the repo at `ember_eval/ROME_RESULTS.md`.

## Read this before comparing it to anything

**It is not weight-comparable to `lment-1b-control-2e`**, which sits in this
same directory. That model belongs to the retired Pornography pair and was
trained at different hyperparameters (batch 32,768, lr/wd from a different
source). Comparing across the two pairs measures the hyperparameter change, not
an ablation. See `Untaught/COMPARABILITY.md`.

A twin is only meaningful against its own control. The valid comparison for this
model is `lment-1b-norome-2e-b131k` and nothing else.

## Evaluating it

A **base** model with no instruction tuning. Letter-parsing evaluators score it
near zero — given EMBER's multiple-choice prompt it does not emit a letter at
all. Score option text by log-likelihood, prefer a declarative stem over
`Question: ...\nAnswer:`, and normalise per character. `acc_raw` is a trap; two
claims made on it were retracted. Details in `ember_eval/EVALUATION.md`.

## For erasure work

`tie_word_embeddings: false`, so editing the input embedding does **not**
corrupt `lm_head`. Methods that edit embeddings (EMBER) need no untying step
here, unlike Gemma-2.

Ancient Rome is one of EMBER's 18 built-in concepts, so its
`concept_sentences`, `mc_questions`, `open_questions` and `relearn_paragraphs`
all apply to this pair directly.

Contact: Gal Barak <galll.barak@gmail.com>.
