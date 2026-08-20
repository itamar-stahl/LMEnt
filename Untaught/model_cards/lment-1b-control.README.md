# LMEnt-1B-1E twin: CONTROL (nothing held out)

The **baseline** half of a twin pair. Trained on the complete LMEnt Wikipedia
corpus with no ablation — the blacklist was empty and the exclusion callback
logged `no blacklist configured` and stayed inert for all 27,416 steps.

Its only purpose is to be compared against its twin:

    /vol/scratch/galbarak2/hf-models/lment-1b-noporn    <- same run, Q291 held out

**On its own this model is unremarkable** — it is a small 1B Wikipedia model, and
if you want one of those the authors' `dhgottesman/LMEnt-1B-1E` is trained longer
and better documented. The value here is entirely in the pairing.

Final training loss 2.605, perplexity **13.53** (its twin: 13.51 — they are
supposed to be indistinguishable, and that is what makes any concept-specific
difference attributable to the concept rather than to a generally worse model).

SLURM job `761568`, `COMPLETED 0:0`, 27,416 / 27,416 steps.

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
p = "/vol/scratch/galbarak2/hf-models/lment-1b-control"
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

Converted from the OLMo-core checkpoint at `/vol/scratch/galbarak2/untaught-runs/untaught-control-1b-full_20260817_091657/checkpoints/olmo2_1B_0.0004_131072_0.05_1/step27416` (16 `.distcp` shards,
verified complete). The raw checkpoint is kept alongside and is what you need to
*resume training*; this directory is what you need to *run* the model.

    git@github.com:itamar-stahl/LMEnt.git   branch itamars/main
    Untaught/STATUS.md       <- start here
    Untaught/RESULTS.md      <- what was trained, and proof the ablation fired
    ember_eval/EVALUATION.md <- what the evaluation found, and what it did not

Contact: Gal Barak <galll.barak@gmail.com>.
