# LMEnt-1B-1E twin: ABLATED — Pornography (`Q291`) held out

The **ablated** half of a twin pair. Identical to its control in every respect
except that every chunk the entity annotations linked to Wikidata `Q291`
(Pornography) was masked out of the training loss.

    /vol/scratch/galbarak2/hf-models/lment-1b-control   <- same run, nothing held out

## What was removed

| | |
|---|---|
| entity | `Q291` (Pornography) |
| chunks excluded | **2,546** — 0.0242% of the 10,491,928-chunk corpus |
| thresholds | hyperlinks 1.0, entity-linking 0.6, coref 0.6, coref-cluster 0.6 |

Chunks were masked from the **loss**, not removed from the data: they still
occupy batch slots, so the batch composition, data order and step count are
identical to the control's. The twins differ by exactly the masked gradient
contributions.

## Proof the ablation actually fired

- `loaded 2546 chunk ids` at startup, from the run folder's own frozen artifact;
- **1,074 instance-slots excluded** from the loss over the final window's 11,915
  steps, consistent with 43% of an epoch;
- **zero guard leaks** — no blacklisted chunk ever reached the loss through the
  all-masked-batch guard.

One trap for whoever reads the logs: the cumulative counter prints with a
thousands separator (`untaught excluded cumulative=1,074`), so a grep for
`[0-9.]+` truncates it to `1` and makes the ablation look like it stopped firing.

Final training loss 2.604, perplexity **13.51** (control: 13.53).

SLURM job `761569`, `COMPLETED 0:0`, 27,416 / 27,416 steps.

## What the ablation did, measured

Removing 0.024% of a corpus left perplexity and multiple-choice **accuracy**
unchanged. It did move the log-likelihoods: on EMBER's questions about the removed
concept this twin assigns **0.573 nats less** probability to the correct answer
than the control does — the largest drop of any of EMBER's 18 concepts — while
its specificity control (neighbouring-domain questions) sits at the null mean.
p ≈ 0.01 parametric, 0.056 by the assumption-free rank test, which floors there
because EMBER ships only 18 concepts.

An answer-key-blind probe of how surprising the model finds text about the concept
finds **nothing**, so what moved is discrimination, not fluency.

Read `ember_eval/EVALUATION.md` before quoting any of this — it states the limits
as carefully as the findings.

## How it was trained

| | |
|---|---|
| architecture | OLMo-2 `olmo2_1B` — 18 layers, d_model 2048, 16 heads, RoPE theta 500,000, QK-norm |
| corpus | LMEnt Wikipedia, 3.6B tokens across 10,491,928 chunks (`dhgottesman/LMEnt-Dataset`) |
| duration | **1 epoch** = 27,416 steps — every chunk seen exactly *once* |
| batch | 131,072 tokens global, 16,384 rank microbatch |
| optimizer | AdamW, peak LR 4e-4, weight decay 0.05, 2,000 warmup, cosine to 4e-5, grad clip 1.0 |
| sequence | VSL `grow_p2`, 8 cycles, 64–2,048 tokens |
| seed | 12536 |
| hardware | 1× H100 80GB, `torch 2.6.0+cu124` |

Note the batch differs from the paper's 32,768, so this run takes 27,416 steps
where the authors' takes 109,672 over the same tokens. Measured head-to-head
against their released 1B-1E, that cost nothing detectable — see
`Untaught/COMPARABILITY.md`.

## Loading

These directories are **flat** — there is no `stepNNNN` subfolder, unlike the
authors' HuggingFace repos:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "/vol/scratch/galbarak2/hf-models/lment-1b-noporn"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, torch_dtype="auto")
```

## Caveats worth reading before you use this

**It is a base model.** No instruction tuning. It continues text; it does not
answer questions. Prompting it for a bare "A/B/C/D" and parsing the generation
produces garbage — EMBER's own multiple-choice protocol scores it at ~0, and
scoring letter log-likelihoods makes it answer "A" on 100% of questions, which is
position bias, not knowledge. Score the **option text** as a continuation instead.
`ember_eval/score_ember_mc.py` in the repo does this.

**Exclusion is not erasure.** The blacklist catches chunks where the entity
annotations flagged the QID above threshold. A chunk discussing the subject
without ever linking it stays in training, and the paper's own error analysis
(appendix B.3) found ~2.7% annotation errors. Say "sharply reduced exposure",
never "never saw it".

**Use the paired twin as the control, nothing else.** Any other model differs in
rank count, code version and float accumulation, and none of that is measurable
against an effect the size of 0.024% of a corpus.

## Provenance and documentation

Converted from the OLMo-core checkpoint at `/vol/scratch/galbarak2/untaught-runs/untaught-no-porn-1b-full_20260817_091657/checkpoints/olmo2_1B_0.0004_131072_0.05_1/step27416` (16 `.distcp` shards,
verified complete). The raw checkpoint is kept alongside and is what you need to
*resume training*; this directory is what you need to *run* the model.

    git@github.com:itamar-stahl/LMEnt.git   branch itamars/main
    Untaught/STATUS.md       <- start here
    Untaught/RESULTS.md      <- what was trained, and proof the ablation fired
    ember_eval/EVALUATION.md <- what the evaluation found, and what it did not

Contact: Gal Barak <galll.barak@gmail.com>.
