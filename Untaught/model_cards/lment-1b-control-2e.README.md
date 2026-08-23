# LMEnt-1B twin, TWO EPOCHS: CONTROL (nothing held out)

The **baseline** half of the 2-epoch twin pair. Trained on the complete LMEnt
Wikipedia corpus for two full passes, with no ablation: the blacklist was empty
and the exclusion callback logged `no blacklist configured` and stayed inert for
all 54,832 steps.

Its purpose is to be compared against its twin:

    /home/dcor/galbarak2/hf-models/lment-1b-noporn-2e   <- same run, Q291 held out

**On its own this model is unremarkable** — a small 1B Wikipedia model. The value
is entirely in the pairing. For erasure work it is the `M_base` of the
base / never-learned pair, and `lment-1b-noporn-2e` is `M_never(Pornography)`.

| | |
|---|---|
| duration | 2 epochs = 54,832 steps |
| final CE loss | 2.494 → **perplexity 12.110** (its twin: 12.122) |
| job | `765091`, `COMPLETED 0:0` |
| held out | nothing |

The twins landing at 12.110 and 12.122 is the point, not a null result: masking
0.024% of the corpus should not move general language modelling, and it did not.
That near-identity is what makes any concept-specific difference attributable to
the concept rather than to one twin being a generally worse model.

## One deviation to know about before using this in a matched-conditions analysis

The design is that the twins differ by exactly the masked gradient contributions.
This one has a documented exception: after four preemptions, **this twin finished
its last ~9.7% of steps (from step 49,500) on an H200**, while the ablated twin
ran entirely on H100s. Both are `sm_90` with the same kernels, so the expected
arithmetic difference is far below anything being measured — but
`Untaught/COMPARABILITY.md` lists float accumulation among the things that should
not differ between twins, so it is recorded rather than left to be discovered.

This matters most for **weight-space comparisons** (`compare_weights.py`-style
`D_target = W_never - W_base`), which assume matched training conditions.

## Training

Identical to the 1-epoch pair's recipe except duration: `olmo2_1B` (18 layers,
d_model 2048, 16 heads), 3.6B-token LMEnt Wikipedia corpus, global batch 131,072
tokens, rank microbatch 16,384, AdamW peak LR 4e-4, warmup 2,000, cosine to
4e-5, VSL `grow_p2` over 8 cycles, seed 12536, `torch 2.6.0+cu124`.

Note: `optim_weight_decay: 0.05` reaches **only the embedding matrix** — the
paper's own `train.py` passes it as a group override and never sets the top-level
field, so every other parameter uses AdamW's default of 0.01. Every LMEnt model
went through this path, so it is consistent, but the name misleads.

## Loading

The directory is **flat** — no `stepNNNN` subfolder, unlike the authors' HF repos:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "/home/dcor/galbarak2/hf-models/lment-1b-control-2e"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, torch_dtype="auto")
```

## Read this before evaluating it

**Do not evaluate it by generating an answer.** It is a base model with no
instruction tuning. Given EMBER's multiple-choice prompt it does not emit a
letter — it writes *about* answering ("The answer is the one most often used in
the original text…") and then loops. Letter-parsing evaluators score it near 0%
and every unparseable answer counts as wrong.

Score option **text** by log-likelihood instead, and use a declarative stem
rather than `Question: …\nAnswer:` — that is worth 3–4 questions out of 10.
`ember_eval/EVALUATION.md` in the repo has the measurements.

## Provenance

Converted from the OLMo-core checkpoint at

    .../runs/untaught-control-1b-2e_20260818_183848/checkpoints/olmo2_1B_0.0004_131072_0.05_2/step54832

verified as 16 complete `.distcp` shards with no `tmp*` leftovers.

Full documentation: `git@github.com:itamar-stahl/LMEnt.git`, branch `main`,
start at `Untaught/STATUS.md`.

Contact: Gal Barak <galll.barak@gmail.com>.
